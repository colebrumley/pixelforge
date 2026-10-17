//! The editor's worker thread (docs/editor.md, Section 3.13): slow computations (tile
//! re-extraction, adjust previews of large selections) each run on a thread of their own so they
//! never block a frame.
//!
//! The main thread submits a job (a snapshot of what it needs) and gets a generation
//! number; the worker always runs the newest job it has, skipping older queued ones, and a
//! finished result whose generation is not the newest submitted is dropped. The frame loop
//! polls for the latest result. The thread ends when its [`Worker`] is dropped and its
//! current job is done (it is not joined, so closing the editor never waits for it).

use std::sync::mpsc::{self, Receiver, Sender};

/// A worker thread running `J -> R` jobs.
pub struct Worker<J: Send + 'static, R: Send + 'static> {
    jobs: Sender<(u64, J)>,
    results: Receiver<(u64, R)>,
    /// The generation of the newest job submitted (0: none yet).
    submitted: u64,
    /// The generation of the newest result received.
    received: u64,
}

impl<J: Send + 'static, R: Send + 'static> Worker<J, R> {
    /// Start the thread. `run` computes a result; `wake` is called after each result is sent
    /// (the GUI requests a repaint there, so the result is picked up without input).
    pub fn spawn(
        name: &str,
        run: impl Fn(J) -> R + Send + 'static,
        wake: impl Fn() + Send + 'static,
    ) -> Worker<J, R> {
        let (job_tx, job_rx) = mpsc::channel::<(u64, J)>();
        let (result_tx, result_rx) = mpsc::channel::<(u64, R)>();
        // Without a thread (spawning failed) jobs are never answered and `busy` stays true.
        let _ = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                while let Ok(mut job) = job_rx.recv() {
                    // Only the newest queued job matters.
                    while let Ok(newer) = job_rx.try_recv() {
                        job = newer;
                    }
                    let (generation, input) = job;
                    if result_tx.send((generation, run(input))).is_err() {
                        break;
                    }
                    wake();
                }
            });
        Worker {
            jobs: job_tx,
            results: result_rx,
            submitted: 0,
            received: 0,
        }
    }

    /// Queue a job; returns its generation. Results of earlier jobs are dropped from now on.
    pub fn submit(&mut self, job: J) -> u64 {
        self.submitted += 1;
        // A worker that died (its thread panicked) never answers; `busy` stays true.
        let _ = self.jobs.send((self.submitted, job));
        self.submitted
    }

    /// The result of the newest job, once it is done (`None` while it runs, and for stale
    /// results).
    pub fn poll(&mut self) -> Option<R> {
        let mut latest = None;
        while let Ok((generation, result)) = self.results.try_recv() {
            if generation == self.submitted {
                self.received = generation;
                latest = Some(result);
            }
        }
        latest
    }

    /// Whether the newest job has not delivered its result yet.
    pub fn busy(&self) -> bool {
        self.received < self.submitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait<R: Send + 'static>(w: &mut Worker<u64, R>) -> R {
        let start = Instant::now();
        loop {
            if let Some(r) = w.poll() {
                return r;
            }
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "worker timed out"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn only_the_newest_generation_is_delivered() {
        let mut w = Worker::spawn("test", |x: u64| x * 10, || {});
        assert!(!w.busy());
        for x in 1..=5 {
            w.submit(x);
        }
        assert!(w.busy());
        assert_eq!(wait(&mut w), 50);
        assert!(!w.busy());
        assert_eq!(w.submit(7), 6);
        assert_eq!(wait(&mut w), 70);
        assert_eq!(w.poll(), None);
    }
}

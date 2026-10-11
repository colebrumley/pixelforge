//! The `thread_count` cases of tests/test_native.py: `parallel::thread_count()` reads
//! `PIXELFORGE_THREADS` and falls back to the available parallelism.
//!
//! Every case that changes the environment is in the one test below, so nothing in this
//! binary reads or writes the variable concurrently; the original value is restored.

use pixelforge::parallel::thread_count;

const VAR: &str = "PIXELFORGE_THREADS";

/// test_thread_count_reads_env and test_thread_count_falls_back_to_available_parallelism.
#[test]
fn thread_count_reads_env_and_falls_back_to_available_parallelism() {
    let saved = std::env::var_os(VAR);

    // test_thread_count_reads_env
    for (value, expected) in [("1", 1), ("7", 7), (" 4 ", 4)] {
        std::env::set_var(VAR, value);
        assert_eq!(thread_count(), expected, "{value:?}");
    }

    // test_thread_count_falls_back_to_available_parallelism
    std::env::remove_var(VAR);
    let default = thread_count();
    assert!(default >= 1);
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    assert_eq!(default, available);
    for value in ["0", "-2", "junk", ""] {
        std::env::set_var(VAR, value);
        assert_eq!(thread_count(), default, "{value:?}");
    }

    match saved {
        Some(v) => std::env::set_var(VAR, v),
        None => std::env::remove_var(VAR),
    }
}

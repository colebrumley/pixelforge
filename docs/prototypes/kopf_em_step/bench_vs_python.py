"""Run the Rust E+M step on the real preprocessed inputs; compare timing and numerical agreement with Python."""
import sys, ctypes, time; sys.path.insert(0, ".")
import numpy as np
from pixelforge import io, preprocess
from pixelforge.config import Config
from pixelforge.downscale import kopf as K
S = "docs/prototypes/kopf_em_step"  # build first: cd docs/prototypes/kopf_em_step && cargo build --release
lib = ctypes.CDLL(f"{S}/target/release/libkopf_rs.dylib")  # .so on Linux
P = ctypes.c_void_p
lib.kopf_em_step.argtypes = [ctypes.c_size_t] * 5 + [P] * 11

def py_em(cube, mask, ho, wo, mu_x, mu_y, cov, nu, sigma, ghost):
    """Python E+M step, lifted verbatim (structure) from kopf.run for iteration 1."""
    hi, wi = mask.shape
    layout = K._Layout(hi, wi, ho, wo); slots = layout.slots
    pos_x, pos_y = layout.x.pos[None, :], layout.y.pos[:, None]
    channels = [np.ascontiguousarray(cube[..., c]) for c in range(3)]
    pixel_mask = None if mask.all() else mask
    gamma = np.empty((len(slots), hi, wi)); dx, dy, quad, work, work2 = (np.empty((hi, wi)) for _ in range(5))
    inv_a, inv_b, inv_c = K._inverse_cov(cov); inv_color = 1.0 / (2.0 * sigma * sigma)
    any_ghost = bool(ghost.any())
    def offsets(cx, cy, jy, jx):
        np.subtract(pos_x, layout.expand(cx, jy, jx), out=dx); np.subtract(pos_y, layout.expand(cy, jy, jx), out=dy)
    for s, (jy, jx) in enumerate(slots):
        offsets(mu_x, mu_y, jy, jx)
        np.multiply(layout.expand(inv_a, jy, jx), dx, out=quad); np.multiply(quad, dx, out=quad)
        np.multiply(layout.expand(inv_b, jy, jx), 2.0, out=work); np.multiply(work, dx, out=work); np.multiply(work, dy, out=work); np.add(quad, work, out=quad)
        np.multiply(layout.expand(inv_c, jy, jx), dy, out=work); np.multiply(work, dy, out=work); np.add(quad, work, out=quad)
        dist = work2
        for c in range(3):
            np.subtract(channels[c], layout.expand(nu[..., c], jy, jx), out=work); work *= work
            if c == 0: dist[...] = work
            else: dist += work
        logw = gamma[s]; np.multiply(quad, -0.5, out=logw); dist *= layout.expand(inv_color, jy, jx); logw -= dist
        logw[~layout.slot_valid(jy, jx, pixel_mask)] = -np.inf
        if any_ghost: logw[layout.expand(ghost, jy, jx)] = -np.inf
    pixel_peak = gamma.max(axis=0); pixel_peak[~np.isfinite(pixel_peak)] = 0.0
    gamma -= pixel_peak; np.exp(gamma, out=gamma); pixel_sum = gamma.sum(axis=0); pixel_sum[pixel_sum == 0] = 1.0; gamma /= pixel_sum
    sums = {n: np.zeros((ho, wo)) for n in ("w", "x", "y", "xx", "xy", "yy", "c0", "c1", "c2")}
    for s, (jy, jx) in enumerate(slots):
        g = gamma[s]; offsets(mu_x, mu_y, jy, jx)
        gx, gy = np.multiply(g, dx, out=quad), np.multiply(g, dy, out=work)
        layout.reduce(g, jy, jx, sums["w"]); layout.reduce(gx, jy, jx, sums["x"]); layout.reduce(gy, jy, jx, sums["y"])
        layout.reduce(np.multiply(gx, dx, out=work2), jy, jx, sums["xx"]); layout.reduce(np.multiply(gx, dy, out=work2), jy, jx, sums["xy"]); layout.reduce(np.multiply(gy, dy, out=work2), jy, jx, sums["yy"])
        for c in range(3): layout.reduce(np.multiply(g, channels[c], out=work2), jy, jx, sums[f"c{c}"])
    return np.stack([sums[n] for n in ("w", "x", "y", "xx", "xy", "yy", "c0", "c1", "c2")], axis=-1)

for path, preset in (("out/sample/portrait.png", "background"), ("out/photo/knight_src.png", "sprite")):
    cfg = Config(preset=preset, method="kopf"); loaded = io.load(path, io.MAX_INPUT_PIXELS)
    pre = preprocess.run(loaded.rgb, loaded.alpha, cfg)
    mask = np.ascontiguousarray(pre.mask); hi, wi = mask.shape; wo, ho = pre.target_width, pre.target_height
    cube = K.to_unit_cube(pre.lab)
    # state after a few Python iterations would be more "typical", but cost is data-independent: use iteration-1 state
    gy_, gx_ = np.meshgrid(np.arange(ho) + 0.5, np.arange(wo) + 0.5, indexing="ij")
    mu_x, mu_y = gx_.copy(), gy_.copy()
    cov = np.zeros((ho, wo, 2, 2)); cov[..., 0, 0] = cov[..., 1, 1] = K.COV_INIT
    nu = np.full((ho, wo, 3), 0.5); sigma = np.full((ho, wo), K.SIGMA_INIT)
    cell = (K.cell_index(hi, ho)[:, None] * wo + K.cell_index(wi, wo)[None, :]).ravel()
    ghost = (np.bincount(cell[mask.ravel()], minlength=ho*wo) * 2 < np.bincount(cell, minlength=ho*wo)).reshape(ho, wo)

    t = time.perf_counter(); ref = py_em(cube, mask, ho, wo, mu_x, mu_y, cov, nu, sigma, ghost); t_py = time.perf_counter() - t
    inv_a, inv_b, inv_c = (np.ascontiguousarray(v) for v in K._inverse_cov(cov)); inv_color = np.ascontiguousarray(1.0 / (2.0 * sigma * sigma))
    ch = np.ascontiguousarray(np.stack([cube[..., c] for c in range(3)])); m8 = mask.astype(np.uint8); g8 = ghost.astype(np.uint8)
    nu_c = np.ascontiguousarray(nu); out = np.zeros((ho, wo, 9))
    def call(threads):
        lib.kopf_em_step(hi, wi, ho, wo, threads, ch.ctypes.data, m8.ctypes.data, mu_x.ctypes.data, mu_y.ctypes.data,
                         inv_a.ctypes.data, inv_b.ctypes.data, inv_c.ctypes.data, nu_c.ctypes.data, inv_color.ctypes.data, g8.ctypes.data, out.ctypes.data)
    print(f"\n{path} {wi}x{hi} -> {wo}x{ho}: python E+M {t_py*1000:.0f} ms")
    for th in (1, 4, 8):
        call(th); ts = []
        for _ in range(5): t = time.perf_counter(); call(th); ts.append(time.perf_counter() - t)
        rel = np.abs(out - ref) / np.maximum(np.abs(ref), 1e-300); rel[ref == 0] = np.abs(out[ref == 0])
        print(f"  rust {th} thread(s): {min(ts)*1000:6.1f} ms  ({t_py/min(ts):5.1f}x)  max rel diff vs python {rel.max():.1e}, exact-equal fraction {np.mean(out == ref):.3f}")
    call(8); a = out.copy(); call(8); print("  rust 8-thread run-to-run bitwise identical:", np.array_equal(a, out))

import sys, ctypes; sys.path.insert(0, ".")
import numpy as np
exec(open(sys.argv[1]).read().split("for path, preset in")[0])   # reuse py_em and lib setup from rs_bench.py
from pixelforge import io, preprocess
from pixelforge.config import Config
from pixelforge.downscale import kopf as K
names = ("w", "x", "y", "xx", "xy", "yy", "c0", "c1", "c2")
for path, preset in (("out/photo/knight_src.png", "sprite"), ("out/sample/portrait.png", "background")):
    cfg = Config(preset=preset, method="kopf"); loaded = io.load(path, io.MAX_INPUT_PIXELS)
    pre = preprocess.run(loaded.rgb, loaded.alpha, cfg)
    mask = np.ascontiguousarray(pre.mask); hi, wi = mask.shape; wo, ho = pre.target_width, pre.target_height
    cube = K.to_unit_cube(pre.lab)
    rng = np.random.default_rng(7)
    gy_, gx_ = np.meshgrid(np.arange(ho) + 0.5, np.arange(wo) + 0.5, indexing="ij")
    for label, jitter in (("symmetric init state", 0.0), ("perturbed state", 1.0)):
        mu_x = gx_ + jitter * rng.uniform(-0.25, 0.25, (ho, wo)); mu_y = gy_ + jitter * rng.uniform(-0.25, 0.25, (ho, wo))
        cov = np.zeros((ho, wo, 2, 2)); cov[..., 0, 0] = cov[..., 1, 1] = K.COV_INIT
        if jitter:
            cov[..., 0, 0] = rng.uniform(0.05, 0.1, (ho, wo)); cov[..., 1, 1] = rng.uniform(0.05, 0.1, (ho, wo)); cov[..., 0, 1] = cov[..., 1, 0] = rng.uniform(-0.02, 0.02, (ho, wo))
        nu = np.full((ho, wo, 3), 0.5) + jitter * rng.uniform(-0.3, 0.3, (ho, wo, 3))
        sigma = np.full((ho, wo), K.SIGMA_INIT) if not jitter else rng.uniform(0.01, 1.0, (ho, wo))
        cell = (K.cell_index(hi, ho)[:, None] * wo + K.cell_index(wi, wo)[None, :]).ravel()
        ghost = (np.bincount(cell[mask.ravel()], minlength=ho*wo) * 2 < np.bincount(cell, minlength=ho*wo)).reshape(ho, wo)
        ref = py_em(cube, mask, ho, wo, mu_x, mu_y, cov, nu, sigma, ghost)
        inv_a, inv_b, inv_c = (np.ascontiguousarray(v) for v in K._inverse_cov(cov)); inv_color = np.ascontiguousarray(1.0 / (2.0 * sigma * sigma))
        ch = np.ascontiguousarray(np.stack([cube[..., c] for c in range(3)])); m8 = mask.astype(np.uint8); g8 = ghost.astype(np.uint8)
        nu_c = np.ascontiguousarray(nu); out = np.zeros((ho, wo, 9)); mu_x = np.ascontiguousarray(mu_x); mu_y = np.ascontiguousarray(mu_y)
        lib.kopf_em_step(hi, wi, ho, wo, 8, ch.ctypes.data, m8.ctypes.data, mu_x.ctypes.data, mu_y.ctypes.data, inv_a.ctypes.data, inv_b.ctypes.data, inv_c.ctypes.data, nu_c.ctypes.data, inv_color.ctypes.data, g8.ctypes.data, out.ctypes.data)
        scaled = [float(np.abs(out[..., i] - ref[..., i]).max() / max(np.abs(ref[..., i]).max(), 1e-300)) for i in range(9)]
        # derived M-step quantities, as kopf uses them
        fed = ref[..., 0] > 1e-12
        der = lambda a: np.stack([a[..., i][fed] / a[..., 0][fed] for i in range(1, 9)], -1)
        dd = np.abs(der(out) - der(ref)).max()
        print(f"{path} [{label}]: max |Δ|/max|sum| per quantity: " + ", ".join(f"{n}={v:.0e}" for n, v in zip(names, scaled)) + f"; derived μ/Σ/ν updates max abs diff {dd:.1e}; fed kernels agree: {np.array_equal(fed, out[...,0] > 1e-12)}")

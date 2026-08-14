"""Taichi usage lab on Colab T4 (no GUI).

Follows official patterns from:
  https://docs.taichi-lang.org/docs/hello_world
  https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm88.py

Run:
    colab --auth=adc exec -s t4 -f experiments/taichi_t4_lab.py
"""

import json
import time
from pathlib import Path

import numpy as np
import taichi as ti
import taichi.math as tm

OUT = Path("/content/taichi_t4")
OUT.mkdir(parents=True, exist_ok=True)

ti.init(arch=ti.cuda)
print("taichi", ti.__version__, "cfg.arch", ti.cfg.arch)

RESULTS = {"arch": str(ti.cfg.arch), "taichi": list(ti.__version__), "runs": {}}


def save_gray(field, path: Path) -> None:
    img = field.to_numpy()
    img = np.clip(img, 0.0, 1.0)
    rgb = np.stack([img, img, img], axis=-1)
    ti.tools.imwrite(rgb, str(path))


# ---------------------------------------------------------------------------
# 1. Official Hello World: Julia set  (@ti.func + @ti.kernel + parallel for)
# ---------------------------------------------------------------------------
N = 512
pixels = ti.field(dtype=float, shape=(N * 2, N))
julia_rgb = ti.Vector.field(3, dtype=ti.u8, shape=(N * 2, N))


@ti.func
def complex_sqr(z):
    return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])


@ti.kernel
def paint(t: float):
    for i, j in pixels:
        c = tm.vec2(-0.8, tm.cos(t) * 0.2)
        z = tm.vec2(i / N - 1, j / N - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


@ti.kernel
def julia_to_rgb():
    for i, j in pixels:
        v = ti.max(0.0, ti.min(1.0, pixels[i, j]))
        julia_rgb[i, j] = ti.Vector(
            [
                ti.cast(40 + 200 * (1.0 - v), ti.u8),
                ti.cast(30 + 90 * v, ti.u8),
                ti.cast(255 * v, ti.u8),
            ]
        )


def run_julia():
    frames = []
    paint(0.0)
    t0 = time.perf_counter()
    n_frames = 12
    for k in range(n_frames):
        paint(k * 0.12)
        julia_to_rgb()
        path = OUT / f"01_julia_t{k:02d}.png"
        ti.tools.imwrite(julia_rgb, str(path))
        frames.append(str(path))
    elapsed = time.perf_counter() - t0
    paint(0.0)
    t1 = time.perf_counter()
    for _ in range(60):
        paint(0.3)
    gpu_ms = (time.perf_counter() - t1) / 60 * 1e3
    RESULTS["runs"]["01_julia_hello_world"] = {
        "resolution": f"{N*2}x{N}",
        "frames": n_frames,
        "render_s": round(elapsed, 3),
        "ms_per_frame_steady": round(gpu_ms, 3),
        "images": frames,
        "pattern": "@ti.func complex_sqr + @ti.kernel paint; outermost for is parallel",
    }
    print("julia", RESULTS["runs"]["01_julia_hello_world"])


# ---------------------------------------------------------------------------
# 2. Official MPM88 (MLS-MPM fluid/elastic blob) — no GUI, rasterize particles
# ---------------------------------------------------------------------------
n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 2e-4
p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)
grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))
mpm_rgb = ti.Vector.field(3, dtype=ti.u8, shape=(512, 512))


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C


@ti.kernel
def mpm_init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]
        J[i] = 1


@ti.kernel
def mpm_draw():
    for i, j in mpm_rgb:
        mpm_rgb[i, j] = ti.Vector([17, 47, 65])
    for p in x:
        px = ti.cast(x[p].x * 511.0, ti.i32)
        py = ti.cast(x[p].y * 511.0, ti.i32)
        if 1 <= px < 511 and 1 <= py < 511:
            col = ti.Vector([6, 213, 215])
            mpm_rgb[px, py] = col
            mpm_rgb[px + 1, py] = col
            mpm_rgb[px, py + 1] = col


def run_mpm():
    mpm_init()
    snapshots = {0, 40, 120, 280}
    frames = []
    n_frames = 280
    t0 = time.perf_counter()
    for f in range(n_frames):
        for _ in range(50):
            substep()
        if f in snapshots:
            mpm_draw()
            path = OUT / f"02_mpm88_{f:04d}.png"
            ti.tools.imwrite(mpm_rgb, str(path))
            frames.append(str(path))
    elapsed = time.perf_counter() - t0
    pos = x.to_numpy()
    RESULTS["runs"]["02_mpm88"] = {
        "particles": n_particles,
        "grid": n_grid,
        "display_frames": n_frames,
        "substeps_per_frame": 50,
        "elapsed_s": round(elapsed, 3),
        "ms_per_display_frame": round(elapsed / n_frames * 1e3, 3),
        "mean_y": round(float(pos[:, 1].mean()), 4),
        "images": frames,
        "pattern": "P2G / grid update / G2P; ti.static ndrange unrolls 3x3 stencil",
    }
    print("mpm88", RESULTS["runs"]["02_mpm88"])


# ---------------------------------------------------------------------------
# 3. Fields + atomic reduction (Python scope vs Taichi scope)
# ---------------------------------------------------------------------------
N_RED = 4_000_000
vals = ti.field(dtype=ti.f32, shape=N_RED)
total = ti.field(dtype=ti.f32, shape=())


@ti.kernel
def fill_vals():
    for i in vals:
        vals[i] = ti.sin(i * 0.001)


@ti.kernel
def reduce_sum():
    total[None] = 0.0
    for i in vals:
        ti.atomic_add(total[None], vals[i])


def run_reduction():
    fill_vals()
    reduce_sum()
    ti_sum = float(total[None])
    np_sum = float(vals.to_numpy().sum())
    t0 = time.perf_counter()
    for _ in range(20):
        reduce_sum()
    ti_ms = (time.perf_counter() - t0) / 20 * 1e3
    arr = vals.to_numpy()
    t1 = time.perf_counter()
    for _ in range(20):
        arr.sum()
    np_ms = (time.perf_counter() - t1) / 20 * 1e3
    RESULTS["runs"]["03_atomic_reduction"] = {
        "n": N_RED,
        "taichi_sum": ti_sum,
        "numpy_sum": np_sum,
        "rel_err": abs(ti_sum - np_sum) / (abs(np_sum) + 1e-9),
        "taichi_ms": round(ti_ms, 3),
        "numpy_ms": round(np_ms, 3),
        "pattern": "ti.atomic_add inside parallel for; field[None] is a 0-D scalar",
    }
    print("reduction", RESULTS["runs"]["03_atomic_reduction"])


def main():
    run_julia()
    run_mpm()
    run_reduction()
    path = OUT / "results.json"
    path.write_text(json.dumps(RESULTS, indent=2, default=str))
    print("wrote", path)
    print(json.dumps(RESULTS, indent=2, default=str))


if __name__ == "__main__":
    main()

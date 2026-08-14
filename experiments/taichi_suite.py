"""Taichi lab experiments for a CPU Colab runtime.

Run on the remote VM:
    colab --auth=adc exec -s <session> -f experiments/taichi_suite.py

Outputs land in /content/taichi_experiments/.
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import taichi as ti

OUT = Path("/content/taichi_experiments")
OUT.mkdir(parents=True, exist_ok=True)

ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=True)

RESULTS: dict[str, object] = {
    "taichi": list(ti.__version__),
    "arch": "cpu/x64",
    "experiments": {},
}


def _record(name: str, payload: dict) -> None:
    RESULTS["experiments"][name] = payload
    print(f"\n=== {name} ===")
    for key, value in payload.items():
        print(f"  {key}: {value}")


def _timeit(fn, warmup: int = 1, repeat: int = 5) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


# ---------------------------------------------------------------------------
# Experiment 1 — kernel correctness (SAXPY)
# ---------------------------------------------------------------------------
N_SAXPY = 4_000_000
x = ti.field(dtype=ti.f32, shape=N_SAXPY)
y = ti.field(dtype=ti.f32, shape=N_SAXPY)
z = ti.field(dtype=ti.f32, shape=N_SAXPY)


@ti.kernel
def saxpy(a: float):
    for i in range(N_SAXPY):
        z[i] = a * x[i] + y[i]


def experiment_saxpy() -> None:
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(N_SAXPY, dtype=np.float32)
    y_np = rng.standard_normal(N_SAXPY, dtype=np.float32)
    a = np.float32(1.75)
    x.from_numpy(x_np)
    y.from_numpy(y_np)

    saxpy(a)
    z_ti = z.to_numpy()
    z_np = a * x_np + y_np
    max_abs = float(np.max(np.abs(z_ti - z_np)))

    t_ti = _timeit(lambda: saxpy(a), warmup=2, repeat=8)
    t_np = _timeit(lambda: np.add(np.multiply(a, x_np, dtype=np.float32), y_np, dtype=np.float32), warmup=2, repeat=8)

    _record(
        "01_saxpy_correctness_and_throughput",
        {
            "n": N_SAXPY,
            "max_abs_error": max_abs,
            "pass": max_abs < 1e-4,
            "taichi_ms": round(t_ti * 1e3, 3),
            "numpy_ms": round(t_np * 1e3, 3),
            "speedup_vs_numpy": round(t_np / t_ti, 3) if t_ti else None,
            "taichi_gflops": round((2 * N_SAXPY) / t_ti / 1e9, 3),
        },
    )


# ---------------------------------------------------------------------------
# Experiment 2 — Mandelbrot (parallel complex iteration + render)
# ---------------------------------------------------------------------------
MAN_W, MAN_H = 1024, 768
MAN_ITERS = 120
mandel = ti.field(dtype=ti.f32, shape=(MAN_W, MAN_H))
mandel_rgb = ti.Vector.field(3, dtype=ti.u8, shape=(MAN_W, MAN_H))


@ti.func
def palette(t: ti.f32) -> ti.math.vec3:
    return ti.math.vec3(
        9.0 * (1.0 - t) * t * t * t,
        15.0 * (1.0 - t) * (1.0 - t) * t * t,
        8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t,
    )


@ti.kernel
def render_mandelbrot(cx: float, cy: float, scale: float):
    for i, j in mandel:
        x0 = cx + (i / MAN_W - 0.5) * scale * (MAN_W / MAN_H)
        y0 = cy + (j / MAN_H - 0.5) * scale
        x, y = 0.0, 0.0
        it = 0
        while x * x + y * y <= 4.0 and it < MAN_ITERS:
            x, y = x * x - y * y + x0, 2.0 * x * y + y0
            it += 1
        t = it / MAN_ITERS
        mandel[i, j] = t
        c = palette(t)
        mandel_rgb[i, j] = ti.Vector(
            [
                ti.cast(ti.min(1.0, c.x) * 255.0, ti.u8),
                ti.cast(ti.min(1.0, c.y) * 255.0, ti.u8),
                ti.cast(ti.min(1.0, c.z) * 255.0, ti.u8),
            ]
        )


def _mandelbrot_numpy(cx, cy, scale):
    xs = cx + (np.linspace(0, 1, MAN_W, dtype=np.float32) - 0.5) * scale * (MAN_W / MAN_H)
    ys = cy + (np.linspace(0, 1, MAN_H, dtype=np.float32) - 0.5) * scale
    c = xs[:, None] + 1j * ys[None, :]
    z = np.zeros_like(c)
    out = np.zeros(c.shape, dtype=np.float32)
    live = np.ones(c.shape, dtype=bool)
    for it in range(MAN_ITERS):
        z[live] = z[live] ** 2 + c[live]
        escaped = live & (np.abs(z) > 2.0)
        out[escaped] = (it + 1) / MAN_ITERS
        live &= ~escaped
        if not live.any():
            break
    out[live] = 1.0
    return out


def experiment_mandelbrot() -> None:
    cx, cy, scale = -0.743643887037151, 0.131825904205330, 0.0025
    render_mandelbrot(cx, cy, scale)
    path = OUT / "02_mandelbrot.png"
    ti.tools.imwrite(mandel_rgb, str(path))

    t_ti = _timeit(lambda: render_mandelbrot(cx, cy, scale), warmup=1, repeat=3)
    t_np = _timeit(lambda: _mandelbrot_numpy(cx, cy, scale), warmup=0, repeat=1)

    _record(
        "02_mandelbrot",
        {
            "resolution": f"{MAN_W}x{MAN_H}",
            "max_iters": MAN_ITERS,
            "center": [cx, cy],
            "taichi_ms": round(t_ti * 1e3, 2),
            "numpy_ms": round(t_np * 1e3, 2),
            "speedup_vs_numpy": round(t_np / t_ti, 2),
            "image": str(path),
        },
    )


# ---------------------------------------------------------------------------
# Experiment 3 — 2D wave equation (finite difference)
# ---------------------------------------------------------------------------
WAVE_N = 384
wave_u = ti.field(dtype=ti.f32, shape=(WAVE_N, WAVE_N))
wave_v = ti.field(dtype=ti.f32, shape=(WAVE_N, WAVE_N))
wave_rgb = ti.Vector.field(3, dtype=ti.u8, shape=(WAVE_N, WAVE_N))


@ti.kernel
def wave_reset():
    for i, j in wave_u:
        wave_u[i, j] = 0.0
        wave_v[i, j] = 0.0
    cx, cy = WAVE_N // 3, WAVE_N // 2
    for i, j in ti.ndrange(WAVE_N, WAVE_N):
        dx = i - cx
        dy = j - cy
        r2 = dx * dx + dy * dy
        if r2 < 80:
            wave_u[i, j] = 1.2 * ti.exp(-r2 / 18.0)
    cx2, cy2 = 2 * WAVE_N // 3, WAVE_N // 3
    for i, j in ti.ndrange(WAVE_N, WAVE_N):
        dx = i - cx2
        dy = j - cy2
        r2 = dx * dx + dy * dy
        if r2 < 50:
            wave_u[i, j] += -0.9 * ti.exp(-r2 / 12.0)


@ti.kernel
def wave_step(c2: float, damping: float):
    for i, j in wave_u:
        if 0 < i < WAVE_N - 1 and 0 < j < WAVE_N - 1:
            lap = (
                wave_u[i - 1, j]
                + wave_u[i + 1, j]
                + wave_u[i, j - 1]
                + wave_u[i, j + 1]
                - 4.0 * wave_u[i, j]
            )
            acc = c2 * lap
            wave_v[i, j] = (wave_v[i, j] + acc) * damping
    for i, j in wave_u:
        if 0 < i < WAVE_N - 1 and 0 < j < WAVE_N - 1:
            wave_u[i, j] += wave_v[i, j]


@ti.kernel
def wave_to_rgb():
    for i, j in wave_u:
        v = wave_u[i, j]
        t = 0.5 + 0.5 * ti.tanh(2.2 * v)
        r = t
        g = 0.25 + 0.55 * t
        b = 1.0 - 0.65 * t
        wave_rgb[i, j] = ti.Vector(
            [
                ti.cast(r * 255.0, ti.u8),
                ti.cast(g * 255.0, ti.u8),
                ti.cast(b * 255.0, ti.u8),
            ]
        )


def experiment_wave() -> None:
    wave_reset()
    frames = []
    n_steps = 420
    snapshot_at = {0, 80, 200, 419}
    t0 = time.perf_counter()
    for step in range(n_steps):
        wave_step(0.18, 0.995)
        if step in snapshot_at:
            wave_to_rgb()
            path = OUT / f"03_wave_step_{step:03d}.png"
            ti.tools.imwrite(wave_rgb, str(path))
            frames.append(str(path))
    elapsed = time.perf_counter() - t0
    energy = float(np.mean(np.abs(wave_u.to_numpy())))
    _record(
        "03_wave_equation",
        {
            "grid": f"{WAVE_N}x{WAVE_N}",
            "steps": n_steps,
            "elapsed_s": round(elapsed, 3),
            "ms_per_step": round(elapsed / n_steps * 1e3, 3),
            "mean_abs_height": round(energy, 5),
            "frames": frames,
        },
    )


# ---------------------------------------------------------------------------
# Experiment 4 — 2D N-body gravity (Barnes-Hut skipped; O(N^2) on CPU)
# ---------------------------------------------------------------------------
N_BODY = 768
pos = ti.Vector.field(2, dtype=ti.f32, shape=N_BODY)
vel = ti.Vector.field(2, dtype=ti.f32, shape=N_BODY)
acc = ti.Vector.field(2, dtype=ti.f32, shape=N_BODY)
body_rgb = ti.Vector.field(3, dtype=ti.u8, shape=(512, 512))


@ti.kernel
def nbody_init():
    for i in range(N_BODY):
        ang = 2.0 * math.pi * i / N_BODY
        r = 0.18 + 0.22 * (i % 17) / 17.0
        pos[i] = ti.Vector([0.5 + r * ti.cos(ang), 0.5 + r * ti.sin(ang)])
        tang = ti.Vector([-ti.sin(ang), ti.cos(ang)])
        vel[i] = tang * (0.55 + 0.15 * (i % 5) / 5.0)
        acc[i] = ti.Vector([0.0, 0.0])


@ti.kernel
def nbody_step(dt: float, g: float, eps: float):
    for i in range(N_BODY):
        a = ti.Vector([0.0, 0.0])
        for j in range(N_BODY):
            d = pos[j] - pos[i]
            dist2 = d.dot(d) + eps
            a += g * d / (dist2 * ti.sqrt(dist2))
        acc[i] = a
    for i in range(N_BODY):
        vel[i] += acc[i] * dt
        pos[i] += vel[i] * dt
        # soft box bounce
        for k in ti.static(range(2)):
            if pos[i][k] < 0.02:
                pos[i][k] = 0.02
                vel[i][k] *= -0.4
            if pos[i][k] > 0.98:
                pos[i][k] = 0.98
                vel[i][k] *= -0.4


@ti.kernel
def nbody_to_rgb():
    for i, j in body_rgb:
        body_rgb[i, j] = ti.Vector([8, 10, 18])
    for p in range(N_BODY):
        x = ti.cast(pos[p].x * 511.0, ti.i32)
        y = ti.cast(pos[p].y * 511.0, ti.i32)
        if 1 <= x < 511 and 1 <= y < 511:
            speed = ti.min(1.0, vel[p].norm() * 1.8)
            col = ti.Vector(
                [
                    ti.cast(80 + 175 * speed, ti.u8),
                    ti.cast(160 - 40 * speed, ti.u8),
                    ti.cast(255 - 120 * speed, ti.u8),
                ]
            )
            for dx, dy in ti.static([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]):
                body_rgb[x + dx, y + dy] = col


def experiment_nbody() -> None:
    nbody_init()
    n_steps = 180
    t0 = time.perf_counter()
    for _ in range(n_steps):
        nbody_step(0.0018, 0.00012, 1e-4)
    elapsed = time.perf_counter() - t0
    nbody_to_rgb()
    path = OUT / "04_nbody.png"
    ti.tools.imwrite(body_rgb, str(path))
    p = pos.to_numpy()
    v = vel.to_numpy()
    _record(
        "04_nbody_gravity",
        {
            "particles": N_BODY,
            "steps": n_steps,
            "elapsed_s": round(elapsed, 3),
            "ms_per_step": round(elapsed / n_steps * 1e3, 3),
            "mean_speed": round(float(np.linalg.norm(v, axis=1).mean()), 5),
            "bbox": [float(p.min()), float(p.max())],
            "image": str(path),
        },
    )


# ---------------------------------------------------------------------------
# Experiment 5 — Julia set parameter sweep (kernel reuse)
# ---------------------------------------------------------------------------
JULIA_N = 512
julia_rgb = ti.Vector.field(3, dtype=ti.u8, shape=(JULIA_N, JULIA_N))


@ti.kernel
def render_julia(cx: float, cy: float):
    for i, j in julia_rgb:
        x = (i / JULIA_N - 0.5) * 3.0
        y = (j / JULIA_N - 0.5) * 3.0
        it = 0
        while x * x + y * y <= 4.0 and it < 80:
            x, y = x * x - y * y + cx, 2.0 * x * y + cy
            it += 1
        t = it / 80.0
        julia_rgb[i, j] = ti.Vector(
            [
                ti.cast(40 + 200 * t, ti.u8),
                ti.cast(20 + 90 * (1.0 - t), ti.u8),
                ti.cast(255 * (1.0 - t * 0.35), ti.u8),
            ]
        )


def experiment_julia_sweep() -> None:
    params = [
        (-0.8, 0.156),
        (-0.4, 0.6),
        (0.285, 0.01),
        (-0.70176, -0.3842),
    ]
    frames = []
    t0 = time.perf_counter()
    for idx, (cx, cy) in enumerate(params):
        render_julia(cx, cy)
        path = OUT / f"05_julia_{idx}.png"
        ti.tools.imwrite(julia_rgb, str(path))
        frames.append({"c": [cx, cy], "image": str(path)})
    elapsed = time.perf_counter() - t0
    _record(
        "05_julia_parameter_sweep",
        {
            "resolution": f"{JULIA_N}x{JULIA_N}",
            "variants": len(params),
            "elapsed_s": round(elapsed, 3),
            "frames": frames,
        },
    )


def main() -> None:
    print("Taichi experiment suite starting")
    print("output dir:", OUT)
    experiment_saxpy()
    experiment_mandelbrot()
    experiment_wave()
    experiment_nbody()
    experiment_julia_sweep()
    summary_path = OUT / "results.json"
    summary_path.write_text(json.dumps(RESULTS, indent=2))
    print("\nWrote", summary_path)
    print(json.dumps(RESULTS, indent=2))


if __name__ == "__main__":
    main()

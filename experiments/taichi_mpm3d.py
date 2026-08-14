"""3D MLS-MPM on Colab T4: snow, sand, and dough in one box.

Headless particle splat. Writes /content/taichi_mpm3d/.
"""

import json
import time
from pathlib import Path

import taichi as ti
import taichi.math as tm

OUT = Path("/content/taichi_mpm3d")
OUT.mkdir(parents=True, exist_ok=True)

ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=4)
print("arch", ti.cfg.arch)

n_grid = 48
dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 1.2e-4
p_vol = (dx * 0.5) ** 3
p_rho = 1.0
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3

# three cubes: snow / sand / dough
side = 12
n_per = side * side * side
n_particles = n_per * 3

x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
Jp = ti.field(dtype=ti.f32, shape=n_particles)
material = ti.field(dtype=ti.i32, shape=n_particles)

grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

W, H = 720, 480
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
zbuf = ti.field(dtype=ti.f32, shape=(W, H))
rgb8 = ti.Vector.field(3, dtype=ti.u8, shape=(W, H))


@ti.kernel
def init():
    for p in range(n_particles):
        mat = p // n_per
        local = p % n_per
        iz = local % side
        iy = (local // side) % side
        ix = local // (side * side)
        origin = tm.vec3(0.18, 0.42, 0.18)
        if mat == 1:
            origin = tm.vec3(0.42, 0.55, 0.38)
        elif mat == 2:
            origin = tm.vec3(0.28, 0.68, 0.58)
        jitter = tm.vec3(ti.random(), ti.random(), ti.random()) * dx * 0.4
        x[p] = origin + tm.vec3(ix, iy, iz) * dx * 0.55 + jitter
        v[p] = tm.vec3(0.0, -0.8, 0.0)
        F[p] = ti.Matrix.identity(ti.f32, 3)
        C[p] = ti.Matrix.zero(ti.f32, 3, 3)
        Jp[p] = 1.0
        material[p] = mat


@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
    for p in x:
        Xp = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx = Xp - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(ti.f32, 3) + dt * C[p]) @ F[p]
        h = 1.0
        mu = 0.0
        la = 0.0
        if material[p] == 0:
            # snow: hardening neo-hookean + plastic volume
            h = ti.exp(10.0 * (1.0 - Jp[p]))
            mu = 5.0e3 * h
            la = 2.0e4 * h
        elif material[p] == 1:
            # sand-ish: high friction, almost no cohesion
            mu = 1.2e4
            la = 2.5e4
        else:
            # dough: soft, sticky, volume preserving
            mu = 8.0e2
            la = 4.0e4
        U, sig, V = ti.svd(F[p])
        J = sig[0, 0] * sig[1, 1] * sig[2, 2]
        for d in ti.static(range(3)):
            sig[d, d] = ti.max(sig[d, d], 1e-4)
        if material[p] == 0:
            for d in ti.static(range(3)):
                sig[d, d] = ti.min(ti.max(sig[d, d], 1.0 - 2.5e-2), 1.0 + 7.5e-3)
            Jp[p] = ti.max(Jp[p] * J / (sig[0, 0] * sig[1, 1] * sig[2, 2]), 0.6)
            F[p] = U @ sig @ V.transpose()
            J = sig[0, 0] * sig[1, 1] * sig[2, 2]
        elif material[p] == 1:
            # Drucker-Prager-lite: clamp volume, shrink shear
            for d in ti.static(range(3)):
                sig[d, d] = ti.max(sig[d, d], 0.2)
            vol = sig[0, 0] * sig[1, 1] * sig[2, 2]
            vol = ti.max(vol, 0.4)
            s = vol ** (1.0 / 3.0)
            mean = (sig[0, 0] + sig[1, 1] + sig[2, 2]) / 3.0
            for d in ti.static(range(3)):
                sig[d, d] = mean + 0.35 * (sig[d, d] - mean)
                sig[d, d] = ti.max(sig[d, d], 0.15)
            F[p] = U @ sig @ V.transpose()
            J = sig[0, 0] * sig[1, 1] * sig[2, 2]
        stress = 2.0 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(
            ti.f32, 3
        ) * la * J * (J - 1.0)
        stress = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (ti.cast(offset, ti.f32) - fx) * dx
            weight = w[i].x * w[j].y * w[k].z
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1.0 / grid_m[i, j, k]) * grid_v[i, j, k]
            grid_v[i, j, k].y -= dt * gravity
            if i < bound and grid_v[i, j, k].x < 0:
                grid_v[i, j, k].x = 0
            if i > n_grid - bound and grid_v[i, j, k].x > 0:
                grid_v[i, j, k].x = 0
            if j < bound and grid_v[i, j, k].y < 0:
                grid_v[i, j, k].y = 0
            if j > n_grid - bound and grid_v[i, j, k].y > 0:
                grid_v[i, j, k].y = 0
            if k < bound and grid_v[i, j, k].z < 0:
                grid_v[i, j, k].z = 0
            if k > n_grid - bound and grid_v[i, j, k].z > 0:
                grid_v[i, j, k].z = 0
    for p in x:
        Xp = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx = Xp - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = tm.vec3(0.0)
        new_C = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = ti.cast(offset, ti.f32) - fx
            g_v = grid_v[base + offset]
            weight = w[i].x * w[j].y * w[k].z
            new_v += weight * g_v
            new_C += 4.0 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]


@ti.kernel
def clear_fb():
    for i, j in pixels:
        gy = j / H
        pixels[i, j] = tm.vec3(0.07, 0.08, 0.10) * (1.0 - gy) + tm.vec3(0.16, 0.17, 0.20) * gy
        zbuf[i, j] = 1e9


@ti.func
def project(p):
    eye = tm.vec3(1.55, 1.05, 1.75)
    target = tm.vec3(0.45, 0.18, 0.45)
    w = (target - eye).normalized()
    right = tm.vec3(0.0, 1.0, 0.0).cross(w).normalized()
    up = w.cross(right)
    cam = p - eye
    z = cam.dot(w)
    ok = 0
    sx = 0
    sy = 0
    if z > 0.05:
        fov = 0.42
        u = cam.dot(right) / (z * fov)
        v = cam.dot(up) / (z * fov * (H / W))
        xx = (u * 0.5 + 0.5) * W
        yy = (0.5 - v * 0.5) * H
        if 1 <= xx < W - 1 and 1 <= yy < H - 1:
            ok = 1
            sx = ti.cast(xx, ti.i32)
            sy = ti.cast(yy, ti.i32)
    return ok, sx, sy, z


@ti.kernel
def draw_box():
    # ground plane hint
    for i, j in pixels:
        pass
    for s in range(2):
        for t in range(40):
            a = t / 39.0
            p0 = tm.vec3(dx * bound, dx * bound, dx * bound)
            p1 = tm.vec3(1.0 - dx * bound, dx * bound, 1.0 - dx * bound)
            p = p0 * (1.0 - a) + tm.vec3(p1.x, p0.y, p0.z) * a
            if s == 1:
                p = p0 * (1.0 - a) + tm.vec3(p0.x, p0.y, p1.z) * a
            ok, sx, sy, z = project(p)
            if ok == 1 and z < zbuf[sx, sy]:
                zbuf[sx, sy] = z
                pixels[sx, sy] = tm.vec3(0.35, 0.36, 0.40)


@ti.kernel
def draw_particles():
    for p in x:
        ok, sx, sy, z = project(x[p])
        if ok == 1:
            col = tm.vec3(0.92, 0.95, 1.0)
            if material[p] == 1:
                col = tm.vec3(0.83, 0.62, 0.28)
            elif material[p] == 2:
                col = tm.vec3(0.78, 0.42, 0.28)
            spd = v[p].norm()
            col = col * (0.55 + 0.45 * ti.min(spd * 0.35, 1.0))
            for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                xx = sx + di
                yy = sy + dj
                if 0 <= xx < W and 0 <= yy < H:
                    if z < zbuf[xx, yy]:
                        zbuf[xx, yy] = z
                        pixels[xx, yy] = col


@ti.kernel
def to_srgb():
    for i, j in pixels:
        c = pixels[i, j]
        g = 1.0 / 2.2
        rgb8[i, j] = ti.Vector(
            [
                ti.cast(ti.min(1.0, ti.pow(ti.max(c.x, 0.0), g)) * 255.0, ti.u8),
                ti.cast(ti.min(1.0, ti.pow(ti.max(c.y, 0.0), g)) * 255.0, ti.u8),
                ti.cast(ti.min(1.0, ti.pow(ti.max(c.z, 0.0), g)) * 255.0, ti.u8),
            ]
        )


def save(name):
    clear_fb()
    draw_box()
    draw_particles()
    to_srgb()
    path = OUT / name
    ti.tools.imwrite(rgb8, str(path))
    return str(path)


def main():
    init()
    n_frames = 180
    snaps = {0, 30, 70, 120, 179}
    images = []
    t0 = time.perf_counter()
    for f in range(n_frames):
        for _ in range(40):
            substep()
        if f in snaps:
            images.append({"frame": f, "path": save(f"mpm3d_{f:03d}.png")})
            print("snap", f)
    elapsed = time.perf_counter() - t0
    t1 = time.perf_counter()
    for _ in range(40):
        substep()
    step_ms = (time.perf_counter() - t1) / 40 * 1e3
    results = {
        "arch": str(ti.cfg.arch),
        "particles": n_particles,
        "grid": n_grid,
        "frames": n_frames,
        "elapsed_s": round(elapsed, 3),
        "ms_per_substep": round(step_ms, 3),
        "materials": {"0": "snow", "1": "sand", "2": "dough"},
        "images": images,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""2D WCSPH dam-break splash + shallow-water rain on Colab T4.

Writes /content/taichi_fluids/.
"""

import json
import time
from pathlib import Path

import taichi as ti
import taichi.math as tm

OUT = Path("/content/taichi_fluids")
OUT.mkdir(parents=True, exist_ok=True)

ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=8)
print("arch", ti.cfg.arch)

# ---- SPH ----
n_sph = 6400
h = 0.018
h2 = h * h
mass = 0.8 * h * h * 1000.0
rest_rho = 1000.0
cs = 28.0
visc = 0.12
gravity = tm.vec2(0.0, -9.8)
dt_sph = 4.0e-4

sph_x = ti.Vector.field(2, dtype=ti.f32, shape=n_sph)
sph_v = ti.Vector.field(2, dtype=ti.f32, shape=n_sph)
sph_rho = ti.field(dtype=ti.f32, shape=n_sph)
sph_p = ti.field(dtype=ti.f32, shape=n_sph)
sph_a = ti.Vector.field(2, dtype=ti.f32, shape=n_sph)

HASH = 64
MAXC = 24
cell_count = ti.field(dtype=ti.i32, shape=HASH * HASH)
cell_part = ti.field(dtype=ti.i32, shape=(HASH * HASH, MAXC))

# ---- shallow water ----
SW = 192
g = 9.8
dx_sw = 1.0 / SW
dt_sw = 0.002
hgt = ti.field(dtype=ti.f32, shape=(SW, SW))
hgt_n = ti.field(dtype=ti.f32, shape=(SW, SW))
hu = ti.field(dtype=ti.f32, shape=(SW, SW))
hv = ti.field(dtype=ti.f32, shape=(SW, SW))
hu_n = ti.field(dtype=ti.f32, shape=(SW, SW))
hv_n = ti.field(dtype=ti.f32, shape=(SW, SW))

W, H = 640, 360
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
rgb8 = ti.Vector.field(3, dtype=ti.u8, shape=(W, H))


@ti.func
def sph_hash(p):
    ix = ti.min(HASH - 1, ti.max(0, ti.cast(p.x * HASH, ti.i32)))
    iy = ti.min(HASH - 1, ti.max(0, ti.cast(p.y * HASH, ti.i32)))
    return ix * HASH + iy


@ti.kernel
def sph_init():
    n_x = 40
    n_y = n_sph // n_x
    for p in range(n_sph):
        ix = p % n_x
        iy = p // n_x
        sph_x[p] = tm.vec2(0.08 + ix * h * 0.55, 0.08 + iy * h * 0.55)
        sph_v[p] = tm.vec2(0.0, 0.0)


@ti.kernel
def sph_build():
    for c in range(HASH * HASH):
        cell_count[c] = 0
    for p in range(n_sph):
        c = sph_hash(sph_x[p])
        s = ti.atomic_add(cell_count[c], 1)
        if s < MAXC:
            cell_part[c, s] = p


@ti.func
def kernel_w(r):
    q = r / h
    val = 0.0
    if q < 1.0:
        val = (1.0 - q) * (1.0 - q) * (1.0 - q)
    return val


@ti.func
def kernel_dw(rvec):
    r = rvec.norm()
    g = tm.vec2(0.0)
    if r > 1e-6 and r < h:
        q = r / h
        g = rvec / r * (-3.0 / h) * (1.0 - q) * (1.0 - q)
    return g


@ti.kernel
def sph_density():
    for i in range(n_sph):
        rho = 0.0
        c0 = sph_hash(sph_x[i])
        ix = c0 // HASH
        iy = c0 % HASH
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x = ti.min(HASH - 1, ti.max(0, ix + di))
            y = ti.min(HASH - 1, ti.max(0, iy + dj))
            c = x * HASH + y
            n = ti.min(cell_count[c], MAXC)
            for s in range(n):
                j = cell_part[c, s]
                r = (sph_x[i] - sph_x[j]).norm()
                rho += mass * kernel_w(r)
        sph_rho[i] = ti.max(rho, rest_rho * 0.3)
        sph_p[i] = cs * cs * (sph_rho[i] - rest_rho)


@ti.kernel
def sph_force():
    for i in range(n_sph):
        acc = gravity
        c0 = sph_hash(sph_x[i])
        ix = c0 // HASH
        iy = c0 % HASH
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x = ti.min(HASH - 1, ti.max(0, ix + di))
            y = ti.min(HASH - 1, ti.max(0, iy + dj))
            c = x * HASH + y
            n = ti.min(cell_count[c], MAXC)
            for s in range(n):
                j = cell_part[c, s]
                if j != i:
                    rij = sph_x[i] - sph_x[j]
                    r = rij.norm()
                    if r < h and r > 1e-6:
                        grad = kernel_dw(rij)
                        acc -= mass * (sph_p[i] / (sph_rho[i] * sph_rho[i]) + sph_p[j] / (sph_rho[j] * sph_rho[j])) * grad
                        vij = sph_v[i] - sph_v[j]
                        acc += visc * mass / sph_rho[j] * vij * kernel_w(r) / (h * h)
        sph_a[i] = acc


@ti.kernel
def sph_integrate():
    for i in range(n_sph):
        sph_v[i] += dt_sph * sph_a[i]
        sph_x[i] += dt_sph * sph_v[i]
        if sph_x[i].x < 0.02:
            sph_x[i].x = 0.02
            sph_v[i].x *= -0.35
        if sph_x[i].x > 0.98:
            sph_x[i].x = 0.98
            sph_v[i].x *= -0.35
        if sph_x[i].y < 0.02:
            sph_x[i].y = 0.02
            sph_v[i].y *= -0.25
        if sph_x[i].y > 0.98:
            sph_x[i].y = 0.98
            sph_v[i].y *= -0.35


@ti.kernel
def draw_sph():
    for i, j in pixels:
        pixels[i, j] = tm.vec3(0.06, 0.08, 0.12)
    for p in range(n_sph):
        sx = ti.cast(sph_x[p].x * (W - 1), ti.i32)
        sy = ti.cast((1.0 - sph_x[p].y) * (H - 1), ti.i32)
        spd = sph_v[p].norm()
        col = tm.vec3(0.15, 0.45, 0.95) * (0.55 + 0.45 * ti.min(spd * 0.15, 1.0))
        col += tm.vec3(0.7, 0.9, 1.0) * ti.min(spd * 0.08, 0.4)
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            x = sx + di
            y = sy + dj
            if 0 <= x < W and 0 <= y < H:
                pixels[x, y] = col


@ti.kernel
def sw_init():
    for i, j in hgt:
        hgt[i, j] = 0.18
        hu[i, j] = 0.0
        hv[i, j] = 0.0


@ti.kernel
def sw_drop(cx: float, cy: float, amp: float):
    for i, j in hgt:
        u = i / SW
        v = j / SW
        d2 = (u - cx) * (u - cx) + (v - cy) * (v - cy)
        hgt[i, j] += amp * ti.exp(-d2 * 420.0)


@ti.kernel
def sw_step():
    for i, j in hgt:
        ip = ti.min(i + 1, SW - 1)
        im = ti.max(i - 1, 0)
        jp = ti.min(j + 1, SW - 1)
        jm = ti.max(j - 1, 0)
        h = ti.max(hgt[i, j], 1e-4)
        dhdx = (hgt[ip, j] - hgt[im, j]) * (0.5 / dx_sw)
        dhdy = (hgt[i, jp] - hgt[i, jm]) * (0.5 / dx_sw)
        dudx = (hu[ip, j] - hu[im, j]) * (0.5 / dx_sw)
        dvdy = (hv[i, jp] - hv[i, jm]) * (0.5 / dx_sw)
        hgt_n[i, j] = hgt[i, j] - dt_sw * (dudx + dvdy)
        hu_n[i, j] = hu[i, j] - dt_sw * (g * h * dhdx)
        hv_n[i, j] = hv[i, j] - dt_sw * (g * h * dhdy)
    for i, j in hgt:
        hgt[i, j] = ti.max(hgt_n[i, j], 0.04)
        hu[i, j] = hu_n[i, j] * 0.998
        hv[i, j] = hv_n[i, j] * 0.998


@ti.kernel
def draw_sw():
    for i, j in pixels:
        u = i / W
        v = 1.0 - j / H
        x = ti.min(SW - 1, ti.max(0, ti.cast(u * SW, ti.i32)))
        y = ti.min(SW - 1, ti.max(0, ti.cast(v * SW, ti.i32)))
        ht = hgt[x, y]
        n = (ht - 0.18) * 6.0
        water = tm.vec3(0.05, 0.22, 0.48) + tm.vec3(0.25, 0.55, 0.85) * ti.max(n, 0.0)
        foam = tm.vec3(0.85, 0.93, 1.0) * ti.max(n - 0.35, 0.0)
        pixels[i, j] = water + foam


@ti.kernel
def to_srgb():
    for i, j in pixels:
        c = pixels[i, j]
        gpow = 1.0 / 2.2
        rgb8[i, j] = ti.Vector(
            [
                ti.cast(ti.min(1.0, ti.pow(ti.max(c.x, 0.0), gpow)) * 255.0, ti.u8),
                ti.cast(ti.min(1.0, ti.pow(ti.max(c.y, 0.0), gpow)) * 255.0, ti.u8),
                ti.cast(ti.min(1.0, ti.pow(ti.max(c.z, 0.0), gpow)) * 255.0, ti.u8),
            ]
        )


def save(name):
    to_srgb()
    path = OUT / name
    ti.tools.imwrite(rgb8, str(path))
    return str(path)


def sph_step():
    sph_build()
    sph_density()
    sph_force()
    sph_integrate()


def main():
    results = {"arch": str(ti.cfg.arch), "images": {}}

    sph_init()
    snaps = {0, 40, 90, 160}
    images = []
    t0 = time.perf_counter()
    n_frames = 161
    for f in range(n_frames):
        for _ in range(12):
            sph_step()
        if f in snaps:
            draw_sph()
            images.append({"frame": f, "path": save(f"sph_{f:03d}.png")})
            print("sph", f)
    results["sph"] = {
        "particles": n_sph,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "images": images,
    }

    sw_init()
    sw_drop(0.35, 0.40, 0.22)
    sw_drop(0.62, 0.55, 0.18)
    sw_drop(0.50, 0.28, 0.14)
    images = []
    t1 = time.perf_counter()
    snaps = {0, 40, 90, 180}
    for f in range(181):
        if f == 50:
            sw_drop(0.72, 0.35, 0.16)
        if f == 110:
            sw_drop(0.28, 0.70, 0.14)
        for _ in range(4):
            sw_step()
        if f in snaps:
            draw_sw()
            images.append({"frame": f, "path": save(f"shallow_{f:03d}.png")})
            print("sw", f)
    results["shallow_water"] = {
        "grid": SW,
        "elapsed_s": round(time.perf_counter() - t1, 3),
        "images": images,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

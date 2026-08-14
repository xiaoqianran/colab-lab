"""XPBD silk on Colab T4 — hang a high-res sheet over a rigid sphere.

Not yarn-level silk. Continuum cloth: anisotropic stretch (warp/weft),
very low bending, air drag, wind. Rigid collider is an analytic sphere.

Writes /content/taichi_silk/.
"""

import json
import math
import time
from pathlib import Path

import taichi as ti
import taichi.math as tm

OUT = Path("/content/taichi_silk")
OUT.mkdir(parents=True, exist_ok=True)

ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=3)
print("arch", ti.cfg.arch)

N = 96
NV = (N + 1) * (N + 1)
DT = 1.0 / 240.0
SUBSTEPS = 2
XPBD_ITERS = 12
GRAVITY = tm.vec3(0.0, -9.8, 0.0)

# Silk-ish: almost inextensible warp/weft, weaker shear, tiny bending.
ALPHA_STRETCH = 4.0e-6
ALPHA_SHEAR = 2.5e-5
ALPHA_BEND = 1.2e-2
DAMPING = 0.02
DRAG = 0.12

SPHERE_C = tm.vec3(0.0, 0.22, 0.0)
SPHERE_R = 0.22

W, H = 960, 540

pos = ti.Vector.field(3, dtype=ti.f32, shape=NV)
prev = ti.Vector.field(3, dtype=ti.f32, shape=NV)
vel = ti.Vector.field(3, dtype=ti.f32, shape=NV)
inv_m = ti.field(dtype=ti.f32, shape=NV)
normal = ti.Vector.field(3, dtype=ti.f32, shape=NV)
warp_dir = ti.Vector.field(3, dtype=ti.f32, shape=NV)

n_stretch = 2 * N * (N + 1)
n_shear = 2 * N * N
n_bend = 2 * (N - 1) * (N + 1)
stretch = ti.Vector.field(2, dtype=ti.i32, shape=n_stretch)
stretch_rest = ti.field(dtype=ti.f32, shape=n_stretch)
shear = ti.Vector.field(2, dtype=ti.i32, shape=n_shear)
shear_rest = ti.field(dtype=ti.f32, shape=n_shear)
bend = ti.Vector.field(2, dtype=ti.i32, shape=n_bend)
bend_rest = ti.field(dtype=ti.f32, shape=n_bend)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
zbuf = ti.field(dtype=ti.f32, shape=(W, H))
rgb8 = ti.Vector.field(3, dtype=ti.u8, shape=(W, H))


@ti.func
def vid(i, j):
    return i * (N + 1) + j


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = vid(i, j)
        u = i / N
        v = j / N
        p = tm.vec3((u - 0.5) * 1.15, 0.68, (v - 0.5) * 1.15)
        pos[k] = p
        prev[k] = p
        vel[k] = tm.vec3(0.0, 0.0, 0.0)
        inv_m[k] = 1.0
    for i, j in ti.ndrange(N, N + 1):
        e = i * (N + 1) + j
        a = vid(i, j)
        b = vid(i + 1, j)
        stretch[e] = ti.Vector([a, b])
        stretch_rest[e] = (pos[a] - pos[b]).norm()
    base = N * (N + 1)
    for i, j in ti.ndrange(N + 1, N):
        e = base + i * N + j
        a = vid(i, j)
        b = vid(i, j + 1)
        stretch[e] = ti.Vector([a, b])
        stretch_rest[e] = (pos[a] - pos[b]).norm()
    for i, j in ti.ndrange(N, N):
        s = 2 * (i * N + j)
        a = vid(i, j)
        b = vid(i + 1, j + 1)
        c = vid(i + 1, j)
        d = vid(i, j + 1)
        shear[s] = ti.Vector([a, b])
        shear_rest[s] = (pos[a] - pos[b]).norm()
        shear[s + 1] = ti.Vector([c, d])
        shear_rest[s + 1] = (pos[c] - pos[d]).norm()
    for i, j in ti.ndrange(N - 1, N + 1):
        bidx = i * (N + 1) + j
        a = vid(i, j)
        c2 = vid(i + 2, j)
        bend[bidx] = ti.Vector([a, c2])
        bend_rest[bidx] = (pos[a] - pos[c2]).norm()
    bbase = (N - 1) * (N + 1)
    for i, j in ti.ndrange(N + 1, N - 1):
        bidx = bbase + i * (N - 1) + j
        a = vid(i, j)
        c2 = vid(i, j + 2)
        bend[bidx] = ti.Vector([a, c2])
        bend_rest[bidx] = (pos[a] - pos[c2]).norm()


@ti.func
def solve_edge(i, j, rest, alpha):
    w0 = inv_m[i]
    w1 = inv_m[j]
    w = w0 + w1
    if w > 0:
        d = pos[j] - pos[i]
        L = d.norm()
        if L > 1e-8:
            n = d / L
            C = L - rest
            at = alpha / (DT * DT)
            dl = -C / (w + at)
            pos[i] -= n * (w0 * dl)
            pos[j] += n * (w1 * dl)


@ti.kernel
def xpbd_stretch():
    for e in range(n_stretch):
        a = stretch[e][0]
        b = stretch[e][1]
        solve_edge(a, b, stretch_rest[e], ALPHA_STRETCH)


@ti.kernel
def xpbd_shear():
    for e in range(n_shear):
        a = shear[e][0]
        b = shear[e][1]
        solve_edge(a, b, shear_rest[e], ALPHA_SHEAR)


@ti.kernel
def xpbd_bend():
    for e in range(n_bend):
        a = bend[e][0]
        b = bend[e][1]
        solve_edge(a, b, bend_rest[e], ALPHA_BEND)


@ti.kernel
def predict(t: float):
    wind = tm.vec3(0.08 * ti.sin(1.3 * t), 0.0, 0.08 * ti.cos(0.9 * t))
    for i in range(NV):
        if inv_m[i] > 0:
            v = vel[i]
            v += DT * GRAVITY
            v += DT * wind * DRAG
            v *= 1.0 - DAMPING
            prev[i] = pos[i]
            pos[i] += DT * v


@ti.kernel
def collide_sphere():
    for i in range(NV):
        if inv_m[i] > 0:
            if pos[i].y < 0.008:
                pos[i].y = 0.008
                if vel[i].y < 0:
                    vel[i].y = 0.0
            d = pos[i] - SPHERE_C
            dist = d.norm()
            min_r = SPHERE_R + 0.01
            if dist < min_r and dist > 1e-6:
                n = d / dist
                pos[i] = SPHERE_C + n * min_r


@ti.kernel
def update_vel():
    for i in range(NV):
        if inv_m[i] > 0:
            vel[i] = (pos[i] - prev[i]) / DT
        else:
            vel[i] = tm.vec3(0.0, 0.0, 0.0)


@ti.kernel
def compute_normals():
    for i in range(NV):
        normal[i] = tm.vec3(0.0, 0.0, 0.0)
        warp_dir[i] = tm.vec3(0.0, 0.0, 0.0)
    for i, j in ti.ndrange(N, N):
        a = vid(i, j)
        b = vid(i + 1, j)
        c = vid(i, j + 1)
        d = vid(i + 1, j + 1)
        n1 = (pos[b] - pos[a]).cross(pos[c] - pos[a])
        n2 = (pos[c] - pos[d]).cross(pos[b] - pos[d])
        normal[a] += n1
        normal[b] += n1 + n2
        normal[c] += n1 + n2
        normal[d] += n2
        w = pos[b] - pos[a]
        warp_dir[a] += w
        warp_dir[b] += w
        warp_dir[c] += w
        warp_dir[d] += w
    for i in range(NV):
        if normal[i].norm() > 1e-8:
            normal[i] = normal[i].normalized()
        else:
            normal[i] = tm.vec3(0.0, 1.0, 0.0)
        if warp_dir[i].norm() > 1e-8:
            warp_dir[i] = warp_dir[i].normalized()
        else:
            warp_dir[i] = tm.vec3(1.0, 0.0, 0.0)


@ti.func
def project(p):
    eye = tm.vec3(1.15, 0.82, 1.35)
    target = tm.vec3(0.0, 0.22, 0.0)
    w = (target - eye).normalized()
    right = tm.vec3(0.0, 1.0, 0.0).cross(w)
    if right.norm() < 1e-6:
        right = tm.vec3(1.0, 0.0, 0.0)
    right = right.normalized()
    up = w.cross(right)
    cam = p - eye
    z = cam.dot(w)
    ok = 0
    sx = 0
    sy = 0
    if z > 0.08:
        fov = 0.62
        u = cam.dot(right) / (z * fov)
        v = cam.dot(up) / (z * fov * (H / W))
        x = (u * 0.5 + 0.5) * W
        y = (0.5 - v * 0.5) * H
        if 1 <= x < W - 1 and 1 <= y < H - 1:
            ok = 1
            sx = ti.cast(x, ti.i32)
            sy = ti.cast(y, ti.i32)
    return ok, sx, sy, z


@ti.func
def silk_shade(p, n, wrp):
    eye = tm.vec3(1.15, 0.82, 1.35)
    V = (eye - p).normalized()
    L = tm.vec3(0.35, 0.85, 0.4).normalized()
    ndl = ti.max(n.dot(L), 0.0)
    ndv = ti.max(n.dot(V), 0.0)
    # Kajiya-Kay-ish anisotropic sheen along warp yarns
    t = wrp
    tl = t.dot(L)
    tv = t.dot(V)
    spec = ti.pow(ti.max(0.0, ti.sqrt(ti.max(0.0, 1.0 - tl * tl)) * ti.sqrt(ti.max(0.0, 1.0 - tv * tv)) - tl * tv), 28.0)
    base = tm.vec3(0.72, 0.08, 0.16)
    rim = ti.pow(1.0 - ndv, 3.0) * 0.35
    return base * (0.12 + 0.78 * ndl) + tm.vec3(1.0, 0.82, 0.70) * spec * 0.85 + tm.vec3(0.95, 0.55, 0.45) * rim


@ti.func
def splat(x, y, z, col):
    if 0 <= x < W and 0 <= y < H:
        if z < zbuf[x, y]:
            zbuf[x, y] = z
            pixels[x, y] = col


@ti.kernel
def clear_fb():
    for i, j in pixels:
        gy = j / H
        pixels[i, j] = tm.vec3(0.10, 0.09, 0.11) * (1.0 - gy) + tm.vec3(0.22, 0.20, 0.24) * gy
        zbuf[i, j] = 1e9


@ti.kernel
def draw_sphere():
    for i, j in pixels:
        eye = tm.vec3(1.15, 0.82, 1.35)
        target = tm.vec3(0.0, 0.22, 0.0)
        ww = (target - eye).normalized()
        right = tm.vec3(0.0, 1.0, 0.0).cross(ww).normalized()
        up = ww.cross(right)
        u = (i + 0.5) / W * 2.0 - 1.0
        v = 1.0 - (j + 0.5) / H * 2.0
        fov = 0.62
        rd = (ww + right * u * fov + up * v * fov * (H / W)).normalized()
        oc = eye - SPHERE_C
        b = oc.dot(rd)
        disc = b * b - oc.dot(oc) + SPHERE_R * SPHERE_R
        if disc > 0:
            t = -b - ti.sqrt(disc)
            if t > 0.05:
                p = eye + rd * t
                n = (p - SPHERE_C).normalized()
                L = tm.vec3(0.35, 0.85, 0.4).normalized()
                ndl = ti.max(n.dot(L), 0.0)
                col = tm.vec3(0.55, 0.56, 0.58) * (0.15 + 0.85 * ndl)
                if t < zbuf[i, j]:
                    zbuf[i, j] = t
                    pixels[i, j] = col


@ti.func
def draw_tri(ia, ib, ic):
    pa = pos[ia]
    pb = pos[ib]
    pc = pos[ic]
    oka, xa, ya, za = project(pa)
    okb, xb, yb, zb = project(pb)
    okc, xc, yc, zc = project(pc)
    if oka == 1 and okb == 1 and okc == 1:
        xmin = ti.max(1, ti.min(xa, ti.min(xb, xc)))
        xmax = ti.min(W - 2, ti.max(xa, ti.max(xb, xc)))
        ymin = ti.max(1, ti.min(ya, ti.min(yb, yc)))
        ymax = ti.min(H - 2, ti.max(ya, ti.max(yb, yc)))
        den = float((yb - yc) * (xa - xc) + (xc - xb) * (ya - yc))
        n = (normal[ia] + normal[ib] + normal[ic])
        if n.norm() > 1e-8:
            n = n.normalized()
        else:
            n = tm.vec3(0.0, 0.0, 1.0)
        wrp = warp_dir[ia]
        col = silk_shade((pa + pb + pc) * (1.0 / 3.0), n, wrp)
        if ti.abs(den) > 1e-3 and xmax >= xmin and ymax >= ymin:
            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    w0 = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / den
                    w1 = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / den
                    w2 = 1.0 - w0 - w1
                    if w0 >= -0.02 and w1 >= -0.02 and w2 >= -0.02:
                        z = w0 * za + w1 * zb + w2 * zc
                        splat(x, y, z, col)


@ti.kernel
def draw_cloth():
    for i, j in ti.ndrange(N, N):
        a = vid(i, j)
        b = vid(i + 1, j)
        c = vid(i, j + 1)
        d = vid(i + 1, j + 1)
        draw_tri(a, b, c)
        draw_tri(b, d, c)


@ti.kernel
def to_srgb():
    for i, j in pixels:
        c = pixels[i, j]
        g = 1.0 / 2.2
        c = tm.vec3(ti.pow(ti.max(c.x, 0.0), g), ti.pow(ti.max(c.y, 0.0), g), ti.pow(ti.max(c.z, 0.0), g))
        rgb8[i, j] = ti.Vector(
            [
                ti.cast(ti.min(1.0, c.x) * 255.0, ti.u8),
                ti.cast(ti.min(1.0, c.y) * 255.0, ti.u8),
                ti.cast(ti.min(1.0, c.z) * 255.0, ti.u8),
            ]
        )


def step(t):
    for _ in range(SUBSTEPS):
        predict(t)
        for _ in range(XPBD_ITERS):
            xpbd_stretch()
            xpbd_shear()
            xpbd_bend()
            collide_sphere()
        update_vel()


def save(name):
    compute_normals()
    clear_fb()
    draw_sphere()
    draw_cloth()
    to_srgb()
    path = OUT / name
    ti.tools.imwrite(rgb8, str(path))
    return str(path)


def main():
    init_mesh()
    n_frames = 220
    snaps = {0, 25, 45, 70, 110, 160, 219}
    images = []
    t0 = time.perf_counter()
    for f in range(n_frames):
        t = f / 60.0
        step(t)
        if f in snaps:
            images.append({"frame": f, "path": save(f"silk_{f:03d}.png")})
            print("snap", f)
    elapsed = time.perf_counter() - t0
    # measure one sim second
    t1 = time.perf_counter()
    for f in range(60):
        step(1.0 + f / 60.0)
    sim_rt = time.perf_counter() - t1
    results = {
        "arch": str(ti.cfg.arch),
        "grid": f"{N}x{N}",
        "vertices": NV,
        "frames": n_frames,
        "elapsed_s": round(elapsed, 3),
        "ms_per_frame": round(elapsed / n_frames * 1e3, 2),
        "realtime_ratio": round(1.0 / sim_rt, 2),
        "images": images,
        "model": "XPBD continuum cloth, not yarn-level silk",
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

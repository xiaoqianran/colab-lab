"""XPBD cloth with spatial-hash self-collision on Colab T4.

Same silk-over-sphere setup as taichi_silk.py, but particles that are not
mesh-adjacent cannot occupy the same volume. That is what makes folds stack
instead of clipping through. Writes /content/taichi_cloth_sc/.
"""

import json
import time
from pathlib import Path

import taichi as ti
import taichi.math as tm

OUT = Path("/content/taichi_cloth_sc")
OUT.mkdir(parents=True, exist_ok=True)

ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=5)
print("arch", ti.cfg.arch)

N = 56
NV = (N + 1) * (N + 1)
DT = 1.0 / 240.0
SUBSTEPS = 2
XPBD_ITERS = 10
GRAVITY = tm.vec3(0.0, -9.8, 0.0)
ALPHA_STRETCH = 5.0e-6
ALPHA_SHEAR = 3.0e-5
ALPHA_BEND = 8.0e-3
DAMPING = 0.018
THICK = 0.014
HASH = 40
MAX_CELL = 12

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

cell_count = ti.field(dtype=ti.i32, shape=HASH * HASH * HASH)
cell_part = ti.field(dtype=ti.i32, shape=(HASH * HASH * HASH, MAX_CELL))

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
zbuf = ti.field(dtype=ti.f32, shape=(W, H))
rgb8 = ti.Vector.field(3, dtype=ti.u8, shape=(W, H))

ORIGIN = tm.vec3(-0.7, -0.05, -0.7)
EXTENT = 1.5


@ti.func
def vid(i, j):
    return i * (N + 1) + j


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = vid(i, j)
        u = i / N
        v = j / N
        p = tm.vec3((u - 0.5) * 1.15, 0.70, (v - 0.5) * 1.15)
        pos[k] = p
        prev[k] = p
        vel[k] = tm.vec3(0.0)
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
        solve_edge(stretch[e][0], stretch[e][1], stretch_rest[e], ALPHA_STRETCH)


@ti.kernel
def xpbd_shear():
    for e in range(n_shear):
        solve_edge(shear[e][0], shear[e][1], shear_rest[e], ALPHA_SHEAR)


@ti.kernel
def xpbd_bend():
    for e in range(n_bend):
        solve_edge(bend[e][0], bend[e][1], bend_rest[e], ALPHA_BEND)


@ti.kernel
def predict(t: float):
    wind = tm.vec3(0.05 * ti.sin(1.1 * t), 0.0, 0.05 * ti.cos(0.8 * t))
    for i in range(NV):
        if inv_m[i] > 0:
            vel[i] += DT * GRAVITY + DT * wind
            vel[i] *= 1.0 - DAMPING
            prev[i] = pos[i]
            pos[i] += DT * vel[i]


@ti.kernel
def collide_sphere():
    for i in range(NV):
        if pos[i].y < 0.006:
            pos[i].y = 0.006
        d = pos[i] - SPHERE_C
        dist = d.norm()
        min_r = SPHERE_R + THICK * 0.7
        if dist < min_r and dist > 1e-6:
            pos[i] = SPHERE_C + d / dist * min_r


@ti.func
def hash_of(p):
    q = (p - ORIGIN) / EXTENT
    ix = ti.min(HASH - 1, ti.max(0, ti.cast(q.x * HASH, ti.i32)))
    iy = ti.min(HASH - 1, ti.max(0, ti.cast(q.y * HASH, ti.i32)))
    iz = ti.min(HASH - 1, ti.max(0, ti.cast(q.z * HASH, ti.i32)))
    return (ix * HASH + iy) * HASH + iz


@ti.kernel
def build_hash():
    for c in range(HASH * HASH * HASH):
        cell_count[c] = 0
    for p in range(NV):
        c = hash_of(pos[p])
        slot = ti.atomic_add(cell_count[c], 1)
        if slot < MAX_CELL:
            cell_part[c, slot] = p


@ti.func
def mesh_near(a, b):
    ia = a // (N + 1)
    ja = a % (N + 1)
    ib = b // (N + 1)
    jb = b % (N + 1)
    di = ia - ib
    dj = ja - jb
    if di < 0:
        di = -di
    if dj < 0:
        dj = -dj
    return di <= 2 and dj <= 2


@ti.kernel
def self_collide():
    for a in range(NV):
        c0 = hash_of(pos[a])
        ix = (c0 // HASH) // HASH
        iy = (c0 // HASH) % HASH
        iz = c0 % HASH
        for di, dj, dk in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
            x = ti.min(HASH - 1, ti.max(0, ix + di))
            y = ti.min(HASH - 1, ti.max(0, iy + dj))
            z = ti.min(HASH - 1, ti.max(0, iz + dk))
            c = (x * HASH + y) * HASH + z
            n = ti.min(cell_count[c], MAX_CELL)
            for s in range(n):
                b = cell_part[c, s]
                if b > a:
                    if mesh_near(a, b) == 0:
                        d = pos[b] - pos[a]
                        dist = d.norm()
                        min_d = THICK * 2.0
                        if dist < min_d and dist > 1e-6:
                            nrm = d / dist
                            corr = 0.5 * (min_d - dist)
                            pos[a] -= nrm * corr
                            pos[b] += nrm * corr


@ti.kernel
def update_vel():
    for i in range(NV):
        vel[i] = (pos[i] - prev[i]) / DT


@ti.kernel
def compute_normals():
    for i in range(NV):
        normal[i] = tm.vec3(0.0)
        warp_dir[i] = tm.vec3(0.0)
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
    right = tm.vec3(0.0, 1.0, 0.0).cross(w).normalized()
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
        xx = (u * 0.5 + 0.5) * W
        yy = (0.5 - v * 0.5) * H
        if 1 <= xx < W - 1 and 1 <= yy < H - 1:
            ok = 1
            sx = ti.cast(xx, ti.i32)
            sy = ti.cast(yy, ti.i32)
    return ok, sx, sy, z


@ti.func
def silk_shade(p, n, wrp):
    eye = tm.vec3(1.15, 0.82, 1.35)
    V = (eye - p).normalized()
    L = tm.vec3(0.35, 0.85, 0.4).normalized()
    ndl = ti.max(n.dot(L), 0.0)
    ndv = ti.max(n.dot(V), 0.0)
    t = wrp
    tl = t.dot(L)
    tv = t.dot(V)
    spec = ti.pow(
        ti.max(0.0, ti.sqrt(ti.max(0.0, 1.0 - tl * tl)) * ti.sqrt(ti.max(0.0, 1.0 - tv * tv)) - tl * tv),
        28.0,
    )
    base = tm.vec3(0.18, 0.42, 0.78)
    rim = ti.pow(1.0 - ndv, 3.0) * 0.35
    return base * (0.12 + 0.78 * ndl) + tm.vec3(1.0, 0.90, 0.80) * spec * 0.85 + tm.vec3(0.55, 0.70, 1.0) * rim


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
            if t > 0.05 and t < zbuf[i, j]:
                p = eye + rd * t
                n = (p - SPHERE_C).normalized()
                L = tm.vec3(0.35, 0.85, 0.4).normalized()
                zbuf[i, j] = t
                pixels[i, j] = tm.vec3(0.55, 0.56, 0.58) * (0.15 + 0.85 * ti.max(n.dot(L), 0.0))


@ti.func
def splat(xx, yy, z, col):
    if 0 <= xx < W and 0 <= yy < H:
        if z < zbuf[xx, yy]:
            zbuf[xx, yy] = z
            pixels[xx, yy] = col


@ti.func
def draw_tri(ia, ib, ic):
    pa, pb, pc = pos[ia], pos[ib], pos[ic]
    oka, xa, ya, za = project(pa)
    okb, xb, yb, zb = project(pb)
    okc, xc, yc, zc = project(pc)
    if oka == 1 and okb == 1 and okc == 1:
        xmin = ti.max(1, ti.min(xa, ti.min(xb, xc)))
        xmax = ti.min(W - 2, ti.max(xa, ti.max(xb, xc)))
        ymin = ti.max(1, ti.min(ya, ti.min(yb, yc)))
        ymax = ti.min(H - 2, ti.max(ya, ti.max(yb, yc)))
        den = float((yb - yc) * (xa - xc) + (xc - xb) * (ya - yc))
        n = normal[ia] + normal[ib] + normal[ic]
        if n.norm() > 1e-8:
            n = n.normalized()
        col = silk_shade((pa + pb + pc) * (1.0 / 3.0), n, warp_dir[ia])
        if ti.abs(den) > 1e-3 and xmax >= xmin and ymax >= ymin:
            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    w0 = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / den
                    w1 = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / den
                    w2 = 1.0 - w0 - w1
                    if w0 >= -0.02 and w1 >= -0.02 and w2 >= -0.02:
                        splat(x, y, w0 * za + w1 * zb + w2 * zc, col)


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
        rgb8[i, j] = ti.Vector(
            [
                ti.cast(ti.min(1.0, ti.pow(ti.max(c.x, 0.0), g)) * 255.0, ti.u8),
                ti.cast(ti.min(1.0, ti.pow(ti.max(c.y, 0.0), g)) * 255.0, ti.u8),
                ti.cast(ti.min(1.0, ti.pow(ti.max(c.z, 0.0), g)) * 255.0, ti.u8),
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
            build_hash()
            self_collide()
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
    n_frames = 200
    snaps = {0, 40, 80, 130, 199}
    images = []
    t0 = time.perf_counter()
    for f in range(n_frames):
        step(f / 60.0)
        if f in snaps:
            images.append({"frame": f, "path": save(f"cloth_{f:03d}.png")})
            print("snap", f)
    elapsed = time.perf_counter() - t0
    t1 = time.perf_counter()
    for f in range(30):
        step(1.0 + f / 60.0)
    sim_rt = (time.perf_counter() - t1) / (30.0 / 60.0)
    results = {
        "arch": str(ti.cfg.arch),
        "grid": f"{N}x{N}",
        "vertices": NV,
        "self_collision": True,
        "thickness": THICK,
        "frames": n_frames,
        "elapsed_s": round(elapsed, 3),
        "ms_per_frame": round(elapsed / n_frames * 1e3, 2),
        "realtime_ratio": round(1.0 / sim_rt, 2),
        "images": images,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""Path tracing + spatio-temporal denoise on Colab T4.

Compare:
  8 spp noisy
  8 spp + A-trous (edge-aware, G-buffer)
  8 frames x 8 spp temporal accumulation
  temporal + A-trous each frame (SVGF-lite)
  48 spp brute-force reference

Same camera. Writes /content/taichi_denoise/.
"""

import json
import math
import time
from pathlib import Path

import taichi as ti
import taichi.math as tm

OUT = Path("/content/taichi_denoise")
OUT.mkdir(parents=True, exist_ok=True)

ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=2)
print("arch", ti.cfg.arch)

W, H = 640, 360
FOV = 0.55
MAX_DEPTH = 4
YAW = 0.42

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
rgb8 = ti.Vector.field(3, dtype=ti.u8, shape=(W, H))
noisy = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
accum = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
tmp = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
albedo_b = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
normal_b = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
depth_b = ti.field(dtype=ti.f32, shape=(W, H))


@ti.func
def saturate(x: float) -> float:
    return ti.max(0.0, ti.min(1.0, x))


@ti.func
def sd_sphere(p, r):
    return p.norm() - r


@ti.func
def sd_box(p, b):
    q = tm.vec3(ti.abs(p.x), ti.abs(p.y), ti.abs(p.z)) - b
    return tm.vec3(ti.max(q.x, 0.0), ti.max(q.y, 0.0), ti.max(q.z, 0.0)).norm() + ti.min(
        ti.max(q.x, ti.max(q.y, q.z)), 0.0
    )


@ti.func
def scene_sdf(p):
    d0 = p.y
    d1 = sd_box(p - tm.vec3(0.0, 0.55, 0.0), tm.vec3(0.55, 0.55, 0.55))
    d2 = sd_sphere(p - tm.vec3(-1.45, 0.48, 0.2), 0.48)
    d3 = sd_sphere(p - tm.vec3(1.5, 0.45, 0.35), 0.45)
    d, mid = d0, 0
    if d1 < d:
        d, mid = d1, 1
    if d2 < d:
        d, mid = d2, 2
    if d3 < d:
        d, mid = d3, 3
    return d, mid


@ti.func
def scene_normal(p):
    e = 1.5e-3
    dx, _ = scene_sdf(p + tm.vec3(e, 0, 0))
    dy, _ = scene_sdf(p + tm.vec3(0, e, 0))
    dz, _ = scene_sdf(p + tm.vec3(0, 0, e))
    d, _ = scene_sdf(p)
    return tm.vec3(dx - d, dy - d, dz - d).normalized()


@ti.func
def albedo_of(mid, p):
    col = tm.vec3(0.22, 0.23, 0.25)
    if mid == 0:
        ix = ti.cast(ti.floor(p.x), ti.i32)
        iz = ti.cast(ti.floor(p.z), ti.i32)
        if (ix + iz) % 2 == 0:
            col = tm.vec3(0.42, 0.43, 0.46)
        else:
            col = tm.vec3(0.16, 0.16, 0.18)
    elif mid == 1:
        col = tm.vec3(0.92, 0.38, 0.12)
    elif mid == 2:
        col = tm.vec3(0.92, 0.90, 0.86)
    elif mid == 3:
        col = tm.vec3(0.85, 0.95, 1.0)
    return col


@ti.func
def sky(rd):
    t = saturate(rd.y * 0.5 + 0.5)
    return tm.vec3(0.55, 0.62, 0.72) * (1.0 - t) + tm.vec3(0.89, 0.91, 0.94) * t


@ti.func
def camera_ray(u, v):
    eye = tm.vec3(ti.sin(YAW) * 6.4, 2.6, ti.cos(YAW) * 6.4)
    target = tm.vec3(0.0, 0.45, 0.0)
    w = (target - eye).normalized()
    right = tm.vec3(0.0, 1.0, 0.0).cross(w).normalized()
    up = w.cross(right)
    rd = (w + right * (2.0 * u - 1.0) * ti.tan(FOV) * (W / H) + up * (1.0 - 2.0 * v) * ti.tan(FOV)).normalized()
    return eye, rd


@ti.func
def hit_sphere(ro, rd, c, r):
    oc = ro - c
    b = oc.dot(rd)
    disc = b * b - oc.dot(oc) + r * r
    t = -1.0
    if disc > 0:
        s = ti.sqrt(disc)
        t0 = -b - s
        t1 = -b + s
        if t0 > 1e-3:
            t = t0
        elif t1 > 1e-3:
            t = t1
    return t


@ti.func
def hit_aabb(ro, rd, bmin, bmax):
    inv = tm.vec3(
        1.0 / (rd.x if ti.abs(rd.x) > 1e-8 else 1e-8),
        1.0 / (rd.y if ti.abs(rd.y) > 1e-8 else 1e-8),
        1.0 / (rd.z if ti.abs(rd.z) > 1e-8 else 1e-8),
    )
    t0 = (bmin - ro) * inv
    t1 = (bmax - ro) * inv
    tmin = tm.vec3(ti.min(t0.x, t1.x), ti.min(t0.y, t1.y), ti.min(t0.z, t1.z))
    tmax = tm.vec3(ti.max(t0.x, t1.x), ti.max(t0.y, t1.y), ti.max(t0.z, t1.z))
    t_enter = ti.max(tmin.x, ti.max(tmin.y, tmin.z))
    t_exit = ti.min(tmax.x, ti.min(tmax.y, tmax.z))
    t = -1.0
    if t_exit > t_enter and t_exit > 1e-3:
        t = t_enter if t_enter > 1e-3 else t_exit
    return t


@ti.func
def aabb_normal(p, bmin, bmax):
    e = 1.2e-3
    n = tm.vec3(0.0, 1.0, 0.0)
    if ti.abs(p.x - bmin.x) < e:
        n = tm.vec3(-1.0, 0.0, 0.0)
    elif ti.abs(p.x - bmax.x) < e:
        n = tm.vec3(1.0, 0.0, 0.0)
    elif ti.abs(p.y - bmin.y) < e:
        n = tm.vec3(0.0, -1.0, 0.0)
    elif ti.abs(p.y - bmax.y) < e:
        n = tm.vec3(0.0, 1.0, 0.0)
    elif ti.abs(p.z - bmin.z) < e:
        n = tm.vec3(0.0, 0.0, -1.0)
    elif ti.abs(p.z - bmax.z) < e:
        n = tm.vec3(0.0, 0.0, 1.0)
    return n


@ti.func
def closest_hit(ro, rd):
    t = 1e8
    mid = -1
    p = ro
    n = tm.vec3(0.0, 1.0, 0.0)
    if ti.abs(rd.y) > 1e-6:
        t0 = -ro.y / rd.y
        if 1e-3 < t0 < t:
            q = ro + rd * t0
            if tm.vec2(q.x, q.z).norm() < 16.0:
                t, mid, p, n = t0, 0, q, tm.vec3(0.0, 1.0, 0.0)
    t1 = hit_aabb(ro, rd, tm.vec3(-0.55, 0.0, -0.55), tm.vec3(0.55, 1.1, 0.55))
    if 0.0 < t1 < t:
        q = ro + rd * t1
        t, mid, p, n = t1, 1, q, aabb_normal(q, tm.vec3(-0.55, 0.0, -0.55), tm.vec3(0.55, 1.1, 0.55))
    t2 = hit_sphere(ro, rd, tm.vec3(-1.45, 0.48, 0.2), 0.48)
    if 0.0 < t2 < t:
        q = ro + rd * t2
        t, mid, p, n = t2, 2, q, (q - tm.vec3(-1.45, 0.48, 0.2)).normalized()
    t3 = hit_sphere(ro, rd, tm.vec3(1.5, 0.45, 0.35), 0.45)
    if 0.0 < t3 < t:
        q = ro + rd * t3
        t, mid, p, n = t3, 3, q, (q - tm.vec3(1.5, 0.45, 0.35)).normalized()
    tL = hit_aabb(ro, rd, tm.vec3(-0.9, 2.55, -0.9), tm.vec3(0.9, 2.62, 0.9))
    if 0.0 < tL < t:
        q = ro + rd * tL
        t, mid, p, n = tL, 5, q, tm.vec3(0.0, -1.0, 0.0)
    hit = 1 if mid >= 0 else 0
    return hit, t, p, n, mid


@ti.func
def orthonormal(n):
    a = tm.vec3(0.0, 1.0, 0.0)
    if ti.abs(n.y) > 0.9:
        a = tm.vec3(1.0, 0.0, 0.0)
    t = a.cross(n).normalized()
    b = n.cross(t)
    return t, b


@ti.func
def cosine_hemisphere(n):
    r1 = ti.random()
    r2 = ti.random()
    phi = 6.2831853 * r1
    x = ti.cos(phi) * ti.sqrt(r2)
    y = ti.sin(phi) * ti.sqrt(r2)
    z = ti.sqrt(1.0 - r2)
    t, b = orthonormal(n)
    return (t * x + b * y + n * z).normalized()


@ti.func
def path_trace(ro0, rd0):
    col = tm.vec3(0.0)
    thr = tm.vec3(1.0)
    ro, rd = ro0, rd0
    for depth in range(MAX_DEPTH):
        hit, _, p, n, mid = closest_hit(ro, rd)
        if hit == 0:
            col += thr * sky(rd) * 0.4
            break
        if n.dot(rd) > 0:
            n = -n
        if mid == 5:
            col += thr * tm.vec3(11.0, 10.5, 9.8)
            break
        albedo = albedo_of(mid, p)
        if mid == 2:
            rd = (rd - 2.0 * n.dot(rd) * n + 0.05 * cosine_hemisphere(n)).normalized()
            thr *= albedo
        elif mid == 3:
            fres = 0.04 + 0.96 * ti.pow(1.0 - saturate(-n.dot(rd)), 5.0)
            if ti.random() < fres:
                rd = (rd - 2.0 * n.dot(rd) * n).normalized()
            else:
                refr = tm.refract(rd, n, 1.0 / 1.5)
                if refr.norm() < 1e-5:
                    rd = (rd - 2.0 * n.dot(rd) * n).normalized()
                else:
                    rd = refr.normalized()
                    thr *= albedo
        else:
            rd = cosine_hemisphere(n)
            thr *= albedo
        ro = p + n * 1.5e-3
        if ti.max(thr.x, ti.max(thr.y, thr.z)) < 0.012:
            break
    return col


@ti.kernel
def render_spp(spp: int):
    for i, j in noisy:
        acc = tm.vec3(0.0)
        for s in range(spp):
            u = (i + ti.random()) / W
            v = (j + ti.random()) / H
            ro, rd = camera_ray(u, v)
            acc += path_trace(ro, rd)
        noisy[i, j] = acc / spp


@ti.kernel
def fill_gbuffer():
    for i, j in albedo_b:
        u = (i + 0.5) / W
        v = (j + 0.5) / H
        ro, rd = camera_ray(u, v)
        hit, t, p, n, mid = closest_hit(ro, rd)
        if hit == 1:
            albedo_b[i, j] = albedo_of(mid, p)
            normal_b[i, j] = n
            depth_b[i, j] = t
        else:
            albedo_b[i, j] = tm.vec3(0.0)
            normal_b[i, j] = tm.vec3(0.0, 1.0, 0.0)
            depth_b[i, j] = 1e3


@ti.kernel
def copy_noisy_to(dst: ti.template()):
    for i, j in dst:
        dst[i, j] = noisy[i, j]


@ti.kernel
def accum_add(n: int):
    k = 1.0 / n
    for i, j in accum:
        accum[i, j] = accum[i, j] * (1.0 - k) + noisy[i, j] * k


@ti.kernel
def zero_accum():
    for i, j in accum:
        accum[i, j] = tm.vec3(0.0)


@ti.func
def edge_w(i, j, x, y):
    w = 1.0
    dn = normal_b[i, j].dot(normal_b[x, y])
    w *= ti.exp(-(1.0 - dn) * 16.0)
    dd = ti.abs(depth_b[i, j] - depth_b[x, y])
    w *= ti.exp(-dd * 2.5)
    da = (albedo_b[i, j] - albedo_b[x, y]).norm()
    w *= ti.exp(-da * 8.0)
    return w


@ti.kernel
def atrous(src: ti.template(), dst: ti.template(), step: int):
    for i, j in dst:
        wsum = 0.0001
        acc = tm.vec3(0.0)
        for di, dj in ti.static(ti.ndrange((-2, 3), (-2, 3))):
            x = ti.min(W - 1, ti.max(0, i + di * step))
            y = ti.min(H - 1, ti.max(0, j + dj * step))
            k = 1.0
            ad = ti.abs(di) + ti.abs(dj)
            if ad == 1:
                k = 0.5
            elif ad == 2:
                k = 0.25
            elif ad >= 3:
                k = 0.125
            w = k * edge_w(i, j, x, y)
            acc += src[x, y] * w
            wsum += w
        dst[i, j] = acc / wsum


@ti.kernel
def to_display(src: ti.template()):
    for i, j in pixels:
        c = src[i, j]
        c = tm.vec3(c.x / (c.x + 1.0), c.y / (c.y + 1.0), c.z / (c.z + 1.0))
        g = 1.0 / 2.2
        pixels[i, j] = tm.vec3(ti.pow(c.x, g), ti.pow(c.y, g), ti.pow(c.z, g))


@ti.kernel
def to_srgb():
    for i, j in pixels:
        c = pixels[i, j]
        rgb8[i, j] = ti.Vector(
            [
                ti.cast(saturate(c.x) * 255.0, ti.u8),
                ti.cast(saturate(c.y) * 255.0, ti.u8),
                ti.cast(saturate(c.z) * 255.0, ti.u8),
            ]
        )


def save_from(src, name):
    to_display(src)
    to_srgb()
    path = OUT / name
    ti.tools.imwrite(rgb8, str(path))
    return str(path)


def atrous_filter(src_is_noisy):
    if src_is_noisy:
        copy_noisy_to(tmp)
    else:
        # copy accum -> tmp via noisy buffer? use pixels as scratch is messy
        copy_accum_to_tmp()
    atrous(tmp, pixels, 1)
    atrous(pixels, tmp, 2)
    atrous(tmp, pixels, 4)
    atrous(pixels, tmp, 8)
    return tmp


@ti.kernel
def copy_accum_to_tmp():
    for i, j in tmp:
        tmp[i, j] = accum[i, j]


def denoise_noisy():
    copy_noisy_to(tmp)
    atrous(tmp, pixels, 1)
    atrous(pixels, tmp, 2)
    atrous(tmp, pixels, 4)
    atrous(pixels, tmp, 8)


def denoise_accum():
    copy_accum_to_tmp()
    atrous(tmp, pixels, 1)
    atrous(pixels, tmp, 2)
    atrous(tmp, pixels, 4)
    atrous(pixels, tmp, 8)


def main():
    fill_gbuffer()
    results = {"arch": str(ti.cfg.arch), "resolution": f"{W}x{H}", "images": {}}

    t0 = time.perf_counter()
    render_spp(8)
    results["images"]["noisy_8spp"] = save_from(noisy, "01_noisy_8spp.png")
    results["t_8spp"] = round(time.perf_counter() - t0, 3)

    t1 = time.perf_counter()
    denoise_noisy()
    results["images"]["spatial_8spp"] = save_from(tmp, "02_spatial_8spp.png")
    results["t_spatial"] = round(time.perf_counter() - t1, 3)

    zero_accum()
    t2 = time.perf_counter()
    n_temp = 8
    for f in range(n_temp):
        render_spp(8)
        accum_add(f + 1)
    results["images"]["temporal_64spp"] = save_from(accum, "03_temporal_64spp.png")
    results["t_temporal"] = round(time.perf_counter() - t2, 3)

    denoise_accum()
    results["images"]["svgf_lite"] = save_from(tmp, "04_svgf_lite.png")

    t3 = time.perf_counter()
    render_spp(48)
    results["images"]["ref_48spp"] = save_from(noisy, "05_ref_48spp.png")
    results["t_48spp"] = round(time.perf_counter() - t3, 3)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

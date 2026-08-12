"""Blender-like 3D studio render in Taichi (headless Colab / CUDA).

Two engines, same scene (default cube + metal/glass spheres + torus + area light):

- Viewport: SDF ray march, three-point lighting, soft shadow (EEVEE-ish)
- Cycles: unidirectional path tracer with Lambert / metal / glass / emission

No GUI. Writes PNG under /content/taichi_3d/.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import taichi as ti
import taichi.math as tm

OUT = Path("/content/taichi_3d")
OUT.mkdir(parents=True, exist_ok=True)

ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=7)
print("arch", ti.cfg.arch)

W, H = 960, 540
FOV = 34.0 * math.pi / 180.0
MAX_DEPTH = 6
SPP = 96

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
rgb8 = ti.Vector.field(3, dtype=ti.u8, shape=(W, H))

CAM_TARGET = tm.vec3(0.0, 0.55, 0.0)
CAM_RADIUS = 8.6
CAM_HEIGHT = 3.15


@ti.func
def saturate(x: float) -> float:
    return ti.max(0.0, ti.min(1.0, x))


@ti.func
def rotate_y(p, ang):
    c, s = ti.cos(ang), ti.sin(ang)
    return tm.vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z)


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
def sd_torus(p, r_major, r_minor):
    q = tm.vec2(tm.vec2(p.x, p.z).norm() - r_major, p.y)
    return q.norm() - r_minor


@ti.func
def sd_plane(p):
    return p.y


@ti.func
def scene_sdf(p):
    # 0 plane, 1 cube, 2 metal sphere, 3 glass sphere, 4 torus
    d0 = sd_plane(p)
    d1 = sd_box(p - tm.vec3(0.0, 0.7, 0.0), tm.vec3(0.7, 0.7, 0.7))
    d2 = sd_sphere(p - tm.vec3(-1.85, 0.55, 0.25), 0.55)
    d3 = sd_sphere(p - tm.vec3(1.95, 0.52, 0.45), 0.52)
    d4 = sd_torus(p - tm.vec3(0.15, 0.22, 1.65), 0.62, 0.18)
    d = d0
    mid = 0
    if d1 < d:
        d, mid = d1, 1
    if d2 < d:
        d, mid = d2, 2
    if d3 < d:
        d, mid = d3, 3
    if d4 < d:
        d, mid = d4, 4
    return d, mid


@ti.func
def scene_normal(p):
    e = 1.5e-3
    dx, _ = scene_sdf(p + tm.vec3(e, 0.0, 0.0))
    dy, _ = scene_sdf(p + tm.vec3(0.0, e, 0.0))
    dz, _ = scene_sdf(p + tm.vec3(0.0, 0.0, e))
    d, _ = scene_sdf(p)
    n = tm.vec3(dx - d, dy - d, dz - d)
    return n.normalized()


@ti.func
def albedo_of(mid, p):
    col = tm.vec3(0.25, 0.55, 0.95)
    if mid == 0:
        ix = ti.cast(ti.floor(p.x), ti.i32)
        iz = ti.cast(ti.floor(p.z), ti.i32)
        if (ix + iz) % 2 == 0:
            col = tm.vec3(0.42, 0.43, 0.46)
        else:
            col = tm.vec3(0.22, 0.23, 0.25)
    elif mid == 1:
        col = tm.vec3(0.92, 0.38, 0.12)
    elif mid == 2:
        col = tm.vec3(0.92, 0.90, 0.86)
    elif mid == 3:
        col = tm.vec3(0.85, 0.95, 1.0)
    return col


@ti.func
def camera_ray(u, v, yaw):
    eye = tm.vec3(
        CAM_TARGET.x + CAM_RADIUS * ti.sin(yaw),
        CAM_HEIGHT,
        CAM_TARGET.z + CAM_RADIUS * ti.cos(yaw),
    )
    w = (CAM_TARGET - eye).normalized()
    right = tm.vec3(0.0, 1.0, 0.0).cross(w)
    if right.norm() < 1e-6:
        right = tm.vec3(1.0, 0.0, 0.0)
    else:
        right = right.normalized()
    up = w.cross(right)
    aspect = W / H
    px = (2.0 * u - 1.0) * ti.tan(FOV * 0.5) * aspect
    py = (1.0 - 2.0 * v) * ti.tan(FOV * 0.5)
    rd = (w + right * px + up * py).normalized()
    return eye, rd


@ti.func
def march(ro, rd):
    t = 0.0
    mid = -1
    hit = 0
    p = ro
    for _ in range(96):
        d, mid = scene_sdf(p)
        if d < 1.2e-3:
            hit = 1
            break
        t += d
        p = ro + rd * t
        if t > 40.0:
            break
    return hit, t, p, mid


@ti.func
def soft_shadow(p, ldir):
    t = 0.04
    res = 1.0
    for _ in range(24):
        q = p + ldir * t
        d, _ = scene_sdf(q)
        res = ti.min(res, 12.0 * d / t)
        t += ti.max(d, 0.03)
        if res < 0.02 or t > 18.0:
            break
    return saturate(res)


@ti.func
def ao(p, n):
    occ = 0.0
    sca = 1.0
    for i in range(5):
        dist = 0.04 + 0.12 * i
        d, _ = scene_sdf(p + n * dist)
        occ += (dist - d) * sca
        sca *= 0.7
    return saturate(1.0 - 1.6 * occ)


@ti.func
def sky(rd):
    t = saturate(rd.y * 0.5 + 0.5)
    return tm.vec3(0.55, 0.62, 0.72) * (1.0 - t) + tm.vec3(0.89, 0.91, 0.94) * t


@ti.func
def shade_viewport(p, n, rd, mid):
    albedo = albedo_of(mid, p)
    key_dir = tm.vec3(0.45, 0.82, 0.35).normalized()
    fill_dir = tm.vec3(-0.55, 0.35, 0.25).normalized()
    rim_dir = tm.vec3(-0.15, 0.25, -0.95).normalized()
    key = tm.vec3(1.0, 0.96, 0.90) * 1.35 * ti.max(n.dot(key_dir), 0.0) * soft_shadow(p, key_dir)
    fill = tm.vec3(0.45, 0.55, 0.75) * 0.35 * ti.max(n.dot(fill_dir), 0.0)
    rim = tm.vec3(0.75, 0.85, 1.0) * 0.45 * ti.pow(saturate(1.0 - n.dot(-rd)), 3.0) * ti.max(
        n.dot(rim_dir), 0.0
    )
    amb = tm.vec3(0.16, 0.18, 0.22) * ao(p, n)
    col = albedo * (key + fill + amb) + rim
    if mid == 2:
        r = rd - 2.0 * n.dot(rd) * n
        col = albedo * 0.25 + sky(r) * 0.75
        col *= 0.35 + 0.65 * soft_shadow(p, key_dir)
    if mid == 3:
        eta = 1.0 / 1.5
        r = tm.refract(rd, n, eta)
        if r.norm() < 1e-5:
            r = rd - 2.0 * n.dot(rd) * n
        else:
            r = r.normalized()
        f = 0.04 + 0.96 * ti.pow(1.0 - saturate(-n.dot(rd)), 5.0)
        col = sky(r) * (1.0 - f) * 0.85 + sky(rd - 2.0 * n.dot(rd) * n) * f
    return col


@ti.kernel
def render_viewport(yaw: float):
    for i, j in pixels:
        u = (i + 0.5) / W
        v = (j + 0.5) / H
        ro, rd = camera_ray(u, v, yaw)
        hit, _, p, mid = march(ro, rd)
        if hit == 1:
            n = scene_normal(p)
            pixels[i, j] = shade_viewport(p, n, rd, mid)
        else:
            pixels[i, j] = sky(rd)


@ti.func
def orthonormal(n):
    a = tm.vec3(1.0, 0.0, 0.0)
    if ti.abs(n.y) < 0.9:
        a = tm.vec3(0.0, 1.0, 0.0)
    t = a.cross(n).normalized()
    b = n.cross(t)
    return t, b


@ti.func
def cosine_hemisphere(n):
    r1 = ti.random()
    r2 = ti.random()
    phi = 2.0 * math.pi * r1
    x = ti.cos(phi) * ti.sqrt(r2)
    y = ti.sin(phi) * ti.sqrt(r2)
    z = ti.sqrt(1.0 - r2)
    t, b = orthonormal(n)
    return (t * x + b * y + n * z).normalized()


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
    e = 1e-3
    n = tm.vec3(0.0, 0.0, 1.0)
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
    return n


@ti.func
def hit_plane_y0(ro, rd):
    t = -1.0
    if ti.abs(rd.y) > 1e-6:
        cand = -ro.y / rd.y
        if cand > 1e-3:
            t = cand
    return t


@ti.func
def hit_torus_march(ro, rd):
    t = 0.02
    hit = -1.0
    for _ in range(64):
        p = ro + rd * t
        d = sd_torus(p - tm.vec3(0.15, 0.22, 1.65), 0.62, 0.18)
        if d < 1.5e-3:
            hit = t
            break
        t += d
        if t > 30.0:
            break
    return hit


@ti.func
def closest_hit(ro, rd):
    t = 1e8
    mid = -1
    p = ro
    n = tm.vec3(0.0, 1.0, 0.0)
    t0 = hit_plane_y0(ro, rd)
    if 0.0 < t0 < t:
        q = ro + rd * t0
        if tm.vec2(q.x, q.z).norm() < 18.0:
            t, mid, p, n = t0, 0, q, tm.vec3(0.0, 1.0, 0.0)
    t1 = hit_aabb(ro, rd, tm.vec3(-0.7, 0.0, -0.7), tm.vec3(0.7, 1.4, 0.7))
    if 0.0 < t1 < t:
        q = ro + rd * t1
        t, mid, p, n = t1, 1, q, aabb_normal(q, tm.vec3(-0.7, 0.0, -0.7), tm.vec3(0.7, 1.4, 0.7))
    t2 = hit_sphere(ro, rd, tm.vec3(-1.85, 0.55, 0.25), 0.55)
    if 0.0 < t2 < t:
        q = ro + rd * t2
        t, mid, p, n = t2, 2, q, (q - tm.vec3(-1.85, 0.55, 0.25)).normalized()
    t3 = hit_sphere(ro, rd, tm.vec3(1.95, 0.52, 0.45), 0.52)
    if 0.0 < t3 < t:
        q = ro + rd * t3
        t, mid, p, n = t3, 3, q, (q - tm.vec3(1.95, 0.52, 0.45)).normalized()
    t4 = hit_torus_march(ro, rd)
    if 0.0 < t4 < t:
        q = ro + rd * t4
        t, mid, p = t4, 4, q
        n = scene_normal(q)
    tL = hit_aabb(ro, rd, tm.vec3(-1.1, 3.15, -1.1), tm.vec3(1.1, 3.22, 1.1))
    if 0.0 < tL < t:
        q = ro + rd * tL
        t, mid, p, n = tL, 5, q, tm.vec3(0.0, -1.0, 0.0)
    hit = 1 if mid >= 0 else 0
    return hit, t, p, n, mid


@ti.func
def path_trace(ro0, rd0):
    col = tm.vec3(0.0)
    thr = tm.vec3(1.0)
    ro, rd = ro0, rd0
    for depth in range(MAX_DEPTH):
        hit, _, p, n, mid = closest_hit(ro, rd)
        if hit == 0:
            col += thr * sky(rd) * 0.35
            break
        if n.dot(rd) > 0:
            n = -n
        if mid == 5:
            col += thr * tm.vec3(12.0, 11.4, 10.5)
            break
        albedo = albedo_of(mid, p)
        if mid == 2:
            rd = (rd - 2.0 * n.dot(rd) * n + 0.06 * cosine_hemisphere(n)).normalized()
            thr *= albedo
        elif mid == 3:
            eta = 1.5
            outward = n
            cosi = saturate(-n.dot(rd))
            entering = rd.dot(n) < 0
            if entering:
                eta = 1.0 / 1.5
            else:
                outward = -n
                cosi = saturate(n.dot(rd))
            r0 = (1.0 - 1.5) / (1.0 + 1.5)
            r0 = r0 * r0
            fres = r0 + (1.0 - r0) * ti.pow(1.0 - cosi, 5.0)
            if ti.random() < fres:
                rd = (rd - 2.0 * n.dot(rd) * n).normalized()
            else:
                refr = tm.refract(rd, outward, eta)
                if refr.norm() < 1e-5:
                    rd = (rd - 2.0 * n.dot(rd) * n).normalized()
                else:
                    rd = refr.normalized()
                    thr *= albedo
        else:
            rd = cosine_hemisphere(n)
            thr *= albedo
        ro = p + n * 1.5e-3
        if ti.max(thr.x, ti.max(thr.y, thr.z)) < 0.01:
            break
    return col


@ti.kernel
def render_cycles(yaw: float, spp: int):
    for i, j in pixels:
        acc = tm.vec3(0.0)
        for s in range(spp):
            u = (i + ti.random()) / W
            v = (j + ti.random()) / H
            ro, rd = camera_ray(u, v, yaw)
            acc += path_trace(ro, rd)
        pixels[i, j] = acc / spp


@ti.kernel
def to_srgb():
    for i, j in pixels:
        c = pixels[i, j]
        c = tm.vec3(c.x / (c.x + 1.0), c.y / (c.y + 1.0), c.z / (c.z + 1.0))
        g = 1.0 / 2.2
        c = tm.vec3(ti.pow(c.x, g), ti.pow(c.y, g), ti.pow(c.z, g))
        rgb8[i, j] = ti.Vector(
            [
                ti.cast(saturate(c.x) * 255.0, ti.u8),
                ti.cast(saturate(c.y) * 255.0, ti.u8),
                ti.cast(saturate(c.z) * 255.0, ti.u8),
            ]
        )


def save(name: str) -> str:
    to_srgb()
    path = OUT / name
    ti.tools.imwrite(rgb8, str(path))
    return str(path)


def main():
    results = {"arch": str(ti.cfg.arch), "resolution": f"{W}x{H}", "images": {}}
    yaw0 = 0.42

    t0 = time.perf_counter()
    render_viewport(yaw0)
    results["images"]["viewport"] = save("01_viewport.png")
    results["viewport_s"] = round(time.perf_counter() - t0, 3)

    turn = []
    t1 = time.perf_counter()
    n_turn = 12
    for k in range(n_turn):
        yaw = yaw0 + k * (2 * math.pi / n_turn)
        render_viewport(yaw)
        turn.append(save(f"02_turn_{k:02d}.png"))
    results["images"]["turntable"] = turn
    results["turntable_s"] = round(time.perf_counter() - t1, 3)

    progressive = []
    t2 = time.perf_counter()
    for spp in (8, 32, SPP):
        render_cycles(yaw0, spp)
        progressive.append({"spp": spp, "path": save(f"03_cycles_spp{spp:03d}.png")})
        print("cycles spp", spp, "done")
    results["images"]["cycles"] = progressive
    results["cycles_s"] = round(time.perf_counter() - t2, 3)
    results["spp"] = SPP

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

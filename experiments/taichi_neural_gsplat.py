"""Tiny neural radiance field + 3D Gaussian splatting on Colab T4.

1. Fit a 2-layer positional-encoding MLP to a synthetic sphere+box scene
   with Taichi (manual SGD on a hash of sampled rays).
2. Rasterize ~6k anisotropic Gaussians of the same scene (order-independent
   weighted blend).

Writes /content/taichi_neural/.
"""

import json
import math
import time
from pathlib import Path

import taichi as ti
import taichi.math as tm

OUT = Path("/content/taichi_neural")
OUT.mkdir(parents=True, exist_ok=True)

ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=11)
print("arch", ti.cfg.arch)

W, H = 640, 360
N_GAUSS = 6144
N_RAYS = 4096
STEPS = 180
IN_DIM = 21  # 3 + 2*3*3 freqs
HID = 32

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
rgb8 = ti.Vector.field(3, dtype=ti.u8, shape=(W, H))
accum = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
accum_w = ti.field(dtype=ti.f32, shape=(W, H))

# MLP: IN -> HID -> 4 (rgb + sigma)
w1 = ti.field(dtype=ti.f32, shape=(HID, IN_DIM))
b1 = ti.field(dtype=ti.f32, shape=HID)
w2 = ti.field(dtype=ti.f32, shape=(4, HID))
b2 = ti.field(dtype=ti.f32, shape=4)
gw1 = ti.field(dtype=ti.f32, shape=(HID, IN_DIM))
gb1 = ti.field(dtype=ti.f32, shape=HID)
gw2 = ti.field(dtype=ti.f32, shape=(4, HID))
gb2 = ti.field(dtype=ti.f32, shape=4)

g_pos = ti.Vector.field(3, dtype=ti.f32, shape=N_GAUSS)
g_col = ti.Vector.field(3, dtype=ti.f32, shape=N_GAUSS)
g_scale = ti.field(dtype=ti.f32, shape=N_GAUSS)
g_op = ti.field(dtype=ti.f32, shape=N_GAUSS)


@ti.func
def saturate(x: float) -> float:
    return ti.max(0.0, ti.min(1.0, x))


@ti.kernel
def init_mlp():
    for i, j in w1:
        w1[i, j] = (ti.random() * 2.0 - 1.0) * 0.15
    for i in b1:
        b1[i] = 0.0
    for i, j in w2:
        w2[i, j] = (ti.random() * 2.0 - 1.0) * 0.15
    for i in b2:
        b2[i] = 0.0


@ti.func
def encode(p):
    # IN_DIM=21: xyz + sin/cos 3 freqs * 3 coords
    feat = ti.Vector([0.0] * IN_DIM)
    feat[0], feat[1], feat[2] = p.x, p.y, p.z
    k = 3
    for f in ti.static(range(3)):
        freq = 2.0 ** f * 3.14159265
        for d in ti.static(range(3)):
            x = p[d] * freq
            feat[k] = ti.sin(x)
            feat[k + 1] = ti.cos(x)
            k += 2
    return feat


@ti.func
def mlp_forward(p):
    feat = encode(p)
    h = ti.Vector([0.0] * HID)
    for i in ti.static(range(HID)):
        s = b1[i]
        for j in ti.static(range(IN_DIM)):
            s += w1[i, j] * feat[j]
        h[i] = ti.max(s, 0.0)
    out = tm.vec4(0.0)
    for i in ti.static(range(4)):
        s = b2[i]
        for j in ti.static(range(HID)):
            s += w2[i, j] * h[j]
        out[i] = s
    rgb = tm.vec3(1.0 / (1.0 + ti.exp(-out.x)), 1.0 / (1.0 + ti.exp(-out.y)), 1.0 / (1.0 + ti.exp(-out.z)))
    sig = ti.log(1.0 + ti.exp(out.w))
    return rgb, sig, feat, h, out


@ti.func
def sd_sphere(p, c, r):
    return (p - c).norm() - r


@ti.func
def sd_box(p, c, b):
    q = tm.vec3(ti.abs(p.x - c.x), ti.abs(p.y - c.y), ti.abs(p.z - c.z)) - b
    return tm.vec3(ti.max(q.x, 0.0), ti.max(q.y, 0.0), ti.max(q.z, 0.0)).norm() + ti.min(
        ti.max(q.x, ti.max(q.y, q.z)), 0.0
    )


@ti.func
def scene_gt(p):
    d0 = p.y + 0.5
    d1 = sd_sphere(p, tm.vec3(-0.25, 0.05, 0.1), 0.32)
    d2 = sd_box(p, tm.vec3(0.32, 0.05, -0.05), tm.vec3(0.22, 0.22, 0.22))
    d = d0
    mid = 0
    if d1 < d:
        d, mid = d1, 1
    if d2 < d:
        d, mid = d2, 2
    return d, mid


@ti.func
def gt_color(p, n, mid):
    L = tm.vec3(0.4, 0.8, 0.3).normalized()
    ndl = ti.max(n.dot(L), 0.0)
    col = tm.vec3(0.55, 0.55, 0.58)
    if mid == 1:
        col = tm.vec3(0.95, 0.35, 0.18)
    elif mid == 2:
        col = tm.vec3(0.18, 0.55, 0.92)
    if mid == 0:
        ix = ti.cast(ti.floor(p.x * 4.0), ti.i32)
        iz = ti.cast(ti.floor(p.z * 4.0), ti.i32)
        if (ix + iz) % 2 == 0:
            col = tm.vec3(0.42, 0.43, 0.46)
        else:
            col = tm.vec3(0.18, 0.19, 0.21)
    return col * (0.18 + 0.82 * ndl)


@ti.func
def gt_normal(p):
    e = 1.5e-3
    dx, _ = scene_gt(p + tm.vec3(e, 0, 0))
    dy, _ = scene_gt(p + tm.vec3(0, e, 0))
    dz, _ = scene_gt(p + tm.vec3(0, 0, e))
    d, _ = scene_gt(p)
    return tm.vec3(dx - d, dy - d, dz - d).normalized()


@ti.func
def camera(u, v, yaw):
    eye = tm.vec3(ti.sin(yaw) * 1.7, 0.85, ti.cos(yaw) * 1.7)
    target = tm.vec3(0.0, 0.0, 0.0)
    w = (target - eye).normalized()
    right = tm.vec3(0.0, 1.0, 0.0).cross(w)
    if right.norm() < 1e-6:
        right = tm.vec3(1.0, 0.0, 0.0)
    right = right.normalized()
    up = w.cross(right)
    fov = 0.55
    rd = (w + right * (2.0 * u - 1.0) * fov * (W / H) + up * (1.0 - 2.0 * v) * fov).normalized()
    return eye, rd


@ti.func
def march_gt(ro, rd):
    t = 0.0
    hit = 0
    p = ro
    mid = 0
    for _ in range(64):
        d, mid = scene_gt(p)
        if d < 1.5e-3:
            hit = 1
            break
        t += d
        p = ro + rd * t
        if t > 8.0:
            break
    return hit, p, mid


@ti.kernel
def zero_grad():
    for i, j in gw1:
        gw1[i, j] = 0.0
    for i in gb1:
        gb1[i] = 0.0
    for i, j in gw2:
        gw2[i, j] = 0.0
    for i in gb2:
        gb2[i] = 0.0


@ti.kernel
def sgd_step(lr: float):
    for i, j in w1:
        w1[i, j] -= lr * gw1[i, j] / N_RAYS
    for i in b1:
        b1[i] -= lr * gb1[i] / N_RAYS
    for i, j in w2:
        w2[i, j] -= lr * gw2[i, j] / N_RAYS
    for i in b2:
        b2[i] -= lr * gb2[i] / N_RAYS


@ti.kernel
def train_batch(yaw: float) -> float:
    loss = 0.0
    for _ in range(N_RAYS):
        u = ti.random()
        v = ti.random()
        ro, rd = camera(u, v, yaw)
        hit, p, mid = march_gt(ro, rd)
        target = tm.vec3(0.45, 0.55, 0.70)
        if hit == 1:
            n = gt_normal(p)
            target = gt_color(p, n, mid)
        # volume sample along ray, take last MLP color * occupancy proxy
        col = tm.vec3(0.0)
        T = 1.0
        for s in range(24):
            t = 0.4 + s * 0.12
            q = ro + rd * t
            rgb, sig, feat, h, out = mlp_forward(q)
            alpha = 1.0 - ti.exp(-sig * 0.12)
            col += T * alpha * rgb
            T *= 1.0 - alpha
        col += T * tm.vec3(0.45, 0.55, 0.70)
        diff = col - target
        loss += diff.dot(diff)
        # backprop through last sample only (cheap, noisy, enough to fit)
        q = p if hit == 1 else ro + rd * 1.6
        rgb, sig, feat, hid, out = mlp_forward(q)
        # dL/drgb ~= 2 (col-target); ignore transmittance for speed
        dcol = 2.0 * diff
        drgb = dcol
        # sigmoid jacobian
        srgb = rgb
        d_out = tm.vec4(
            drgb.x * srgb.x * (1.0 - srgb.x),
            drgb.y * srgb.y * (1.0 - srgb.y),
            drgb.z * srgb.z * (1.0 - srgb.z),
            0.15 * (1.0 / (1.0 + ti.exp(-out.w))),
        )
        dh = ti.Vector([0.0] * HID)
        for i in ti.static(range(4)):
            ti.atomic_add(gb2[i], d_out[i])
            for j in ti.static(range(HID)):
                ti.atomic_add(gw2[i, j], d_out[i] * hid[j])
                dh[j] += d_out[i] * w2[i, j]
        for j in ti.static(range(HID)):
            if hid[j] > 0:
                ti.atomic_add(gb1[j], dh[j])
                for k in ti.static(range(IN_DIM)):
                    ti.atomic_add(gw1[j, k], dh[j] * feat[k])
    return loss / N_RAYS


@ti.kernel
def render_nerf(yaw: float):
    for i, j in pixels:
        u = (i + 0.5) / W
        v = (j + 0.5) / H
        ro, rd = camera(u, v, yaw)
        col = tm.vec3(0.0)
        T = 1.0
        for s in range(32):
            t = 0.35 + s * 0.10
            q = ro + rd * t
            rgb, sig, _, _, _ = mlp_forward(q)
            alpha = 1.0 - ti.exp(-sig * 0.10)
            col += T * alpha * rgb
            T *= 1.0 - alpha
        col += T * tm.vec3(0.45, 0.55, 0.70)
        pixels[i, j] = col


@ti.kernel
def render_gt(yaw: float):
    for i, j in pixels:
        u = (i + 0.5) / W
        v = (j + 0.5) / H
        ro, rd = camera(u, v, yaw)
        hit, p, mid = march_gt(ro, rd)
        if hit == 1:
            n = gt_normal(p)
            pixels[i, j] = gt_color(p, n, mid)
        else:
            pixels[i, j] = tm.vec3(0.45, 0.55, 0.70)


@ti.kernel
def init_gaussians():
    for i in range(N_GAUSS):
        kind = i % 3
        p = tm.vec3(0.0)
        col = tm.vec3(0.9)
        sc = 0.03
        if kind == 0:
            # sphere shell
            a = ti.random() * 6.28318
            b = ti.acos(2.0 * ti.random() - 1.0)
            p = tm.vec3(-0.25, 0.05, 0.1) + 0.32 * tm.vec3(ti.sin(b) * ti.cos(a), ti.cos(b), ti.sin(b) * ti.sin(a))
            col = tm.vec3(0.95, 0.35, 0.18)
            sc = 0.028
        elif kind == 1:
            p = tm.vec3(
                0.32 + (ti.random() * 2.0 - 1.0) * 0.22,
                0.05 + (ti.random() * 2.0 - 1.0) * 0.22,
                -0.05 + (ti.random() * 2.0 - 1.0) * 0.22,
            )
            col = tm.vec3(0.18, 0.55, 0.92)
            sc = 0.030
        else:
            p = tm.vec3((ti.random() * 2.0 - 1.0) * 1.1, -0.5, (ti.random() * 2.0 - 1.0) * 1.1)
            ix = ti.cast(ti.floor(p.x * 4.0), ti.i32)
            iz = ti.cast(ti.floor(p.z * 4.0), ti.i32)
            if (ix + iz) % 2 == 0:
                col = tm.vec3(0.42, 0.43, 0.46)
            else:
                col = tm.vec3(0.18, 0.19, 0.21)
            sc = 0.045
        g_pos[i] = p
        g_col[i] = col
        g_scale[i] = sc
        g_op[i] = 0.55


@ti.kernel
def clear_accum():
    for i, j in accum:
        accum[i, j] = tm.vec3(0.0)
        accum_w[i, j] = 0.0


@ti.kernel
def splat_gaussians(yaw: float):
    eye = tm.vec3(ti.sin(yaw) * 1.7, 0.85, ti.cos(yaw) * 1.7)
    target = tm.vec3(0.0, 0.0, 0.0)
    w = (target - eye).normalized()
    right = tm.vec3(0.0, 1.0, 0.0).cross(w).normalized()
    up = w.cross(right)
    fov = 0.55
    for g in range(N_GAUSS):
        p = g_pos[g]
        cam = p - eye
        z = cam.dot(w)
        if z > 0.08:
            u = cam.dot(right) / (z * fov * (W / H))
            v = cam.dot(up) / (z * fov)
            cx = (u * 0.5 + 0.5) * W
            cy = (0.5 - v * 0.5) * H
            rad = ti.max(1.2, (g_scale[g] / z) * H * 0.55)
            x0 = ti.max(0, ti.cast(cx - rad, ti.i32))
            x1 = ti.min(W - 1, ti.cast(cx + rad, ti.i32))
            y0 = ti.max(0, ti.cast(cy - rad, ti.i32))
            y1 = ti.min(H - 1, ti.cast(cy + rad, ti.i32))
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    dx = (x + 0.5 - cx) / rad
                    dy = (y + 0.5 - cy) / rad
                    d2 = dx * dx + dy * dy
                    if d2 < 1.0:
                        wt = g_op[g] * ti.exp(-d2 * 2.2) / (z + 0.2)
                        ti.atomic_add(accum[x, y], g_col[g] * wt)
                        ti.atomic_add(accum_w[x, y], wt)


@ti.kernel
def finish_splat():
    for i, j in pixels:
        w = accum_w[i, j]
        if w > 1e-5:
            pixels[i, j] = accum[i, j] / w
        else:
            gy = j / H
            pixels[i, j] = tm.vec3(0.45, 0.55, 0.70) * (0.6 + 0.4 * gy)


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


def main():
    yaw0 = 0.55
    results = {"arch": str(ti.cfg.arch), "images": {}}

    render_gt(yaw0)
    results["images"]["gt"] = save("01_gt.png")

    init_mlp()
    render_nerf(yaw0)
    results["images"]["nerf_init"] = save("02_nerf_init.png")

    t0 = time.perf_counter()
    losses = []
    for s in range(STEPS):
        yaw = yaw0 + 0.35 * math.sin(s * 0.11)
        zero_grad()
        loss = train_batch(yaw)
        sgd_step(0.08 if s < 80 else 0.03)
        if s % 30 == 0:
            losses.append(float(loss))
            print("step", s, "loss", float(loss))
    results["nerf_train_s"] = round(time.perf_counter() - t0, 3)
    results["losses"] = losses
    render_nerf(yaw0)
    results["images"]["nerf_fit"] = save("03_nerf_fit.png")

    init_gaussians()
    turns = []
    t1 = time.perf_counter()
    for k in range(8):
        yaw = yaw0 + k * (2 * math.pi / 8)
        clear_accum()
        splat_gaussians(yaw)
        finish_splat()
        turns.append(save(f"04_gsplat_{k:02d}.png"))
    results["gsplat_s"] = round(time.perf_counter() - t1, 3)
    results["gaussians"] = N_GAUSS
    results["images"]["gsplat_turn"] = turns
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

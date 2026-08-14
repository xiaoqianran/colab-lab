# Taichi experiments (Colab CPU)

Ran on Colab session `866d93` (CPU / x64) with **Taichi 1.7.4**.

Install and run:

```bash
colab --auth=adc install -s 866d93 taichi numpy
colab --auth=adc exec -s 866d93 -f experiments/taichi_suite.py
```

Outputs: `/content/taichi_experiments/` (copies in `experiments/outputs/`).

## Experiments

| ID | Name | Result |
| --- | --- | --- |
| 01 | SAXPY (4e6) | max abs error **0**; Taichi **5.5 ms** vs NumPy **13.9 ms** (**2.54×**) |
| 02 | Mandelbrot 1024×768 | Taichi **132 ms** vs NumPy **851 ms** (**6.42×**); seahorse-valley zoom |
| 03 | 2D wave 384² × 420 steps | **2.16 ms/step**; two Gaussian drops → expanding rings |
| 04 | N-body 768 particles × 180 | **1.42 ms/step**; orbiting ring stays bounded |
| 05 | Julia sweep 512² × 4 | **0.36 s** for four `c` values |

## Notes

- Runtime is CPU-only; kernels still beat vectorized NumPy on nested iteration (Mandelbrot).
- `from __future__ import annotations` breaks Taichi 1.7 kernel type hints (annotations become strings).
- Kernel argument types must be Python `float`, not `ti.f32`.

## T4 physics / graphics suite

Run on a Colab T4, then **stop the session immediately**:

```bash
colab --auth=adc new -s t4-lab --gpu T4
colab --auth=adc install -s t4-lab taichi numpy
colab --auth=adc exec -s t4-lab -f experiments/taichi_mpm3d.py --timeout 600
colab --auth=adc exec -s t4-lab -f experiments/taichi_cloth_selfcol.py --timeout 600
colab --auth=adc exec -s t4-lab -f experiments/taichi_fluids.py --timeout 600
colab --auth=adc exec -s t4-lab -f experiments/taichi_neural_gsplat.py --timeout 600
colab --auth=adc exec -s t4-lab -f experiments/taichi_pt_denoise.py --timeout 900
colab --auth=adc stop -s t4-lab
```

| Script | What | T4 result |
| --- | --- | --- |
| `taichi_mpm3d.py` | 3D MLS-MPM snow / sand / dough | 5184 particles, 48³, 180 frames, **19.25 s**, 0.052 ms/substep |
| `taichi_cloth_selfcol.py` | XPBD cloth + spatial-hash self-collision | 56×56, 3249 verts, 200 frames, **8.69 s**, 2.56× realtime |
| `taichi_fluids.py` | WCSPH dam break + shallow-water rain | SPH 6400 **2.69 s**; shallow 192² **0.31 s** |
| `taichi_neural_gsplat.py` | Tiny PE-MLP neural field + ~6k Gaussian splats | 6144 Gaussians, splat turntable **0.51 s** (MLP train ~2 min) |
| `taichi_pt_denoise.py` | Path tracing 8 spp vs A-trous / temporal / SVGF-lite vs 48 spp | 640×360; 8 spp **1.62 s**; A-trous **4.33 s**; temporal **0.20 s** |

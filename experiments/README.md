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

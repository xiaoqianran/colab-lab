# Taichi experiments (Colab CPU)

Suite run on a Colab CPU VM via `google-colab-cli`. Taichi 1.7.4, arch `x64`.

## Experiments

| ID | Name | What it tests |
| --- | --- | --- |
| 01 | SAXPY | Kernel correctness vs NumPy, parallel throughput |
| 02 | Mandelbrot | Nested iteration over a 1024×768 field, render to PNG |
| 03 | 2D wave equation | Finite-difference PDE, two Gaussian drops, 4 snapshots |
| 04 | N-body gravity | 768 particles, O(N²) force, soft box bounce |
| 05 | Julia sweep | Reuse one kernel across 4 complex parameters |

## Run

```bash
colab --auth=adc install -s 866d93 taichi matplotlib pillow numpy
colab --auth=adc exec -s 866d93 -f experiments/taichi_suite.py
colab --auth=adc download -s 866d93 /content/taichi_experiments ./taichi_experiments
```

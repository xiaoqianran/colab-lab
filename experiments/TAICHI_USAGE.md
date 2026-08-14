# Taichi usage (what we actually ran on Colab)

Official model, from [Hello, World](https://docs.taichi-lang.org/docs/hello_world) and [MPM88](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm88.py):

1. `ti.init(arch=ti.cuda)` — pick backend (`ti.gpu` tries CUDA → Vulkan → OpenGL).
2. `ti.field` / `ti.Vector.field` / `ti.Matrix.field` — device arrays, like ndarray.
3. `@ti.kernel` — entry point, args must be type-hinted (`float`, not `ti.f32` on 1.7).
4. `@ti.func` — device helper, only callable from kernels/funcs.
5. Outermost `for` in a kernel is **automatically parallel**. Nested loops are serial per thread.
6. `ti.static(...)` unrolls small loops (MPM 3×3 stencil).
7. Colab has no display: skip `ti.GUI`, write PNG with `ti.tools.imwrite`.

Do **not** use `from __future__ import annotations` — Taichi 1.7 needs real types, not strings.

# Kaggle: DiffusionGemma on dual T4

Kaggle notebook for `google/diffusiongemma` (26B-A4B-it) on **2× Tesla T4**.

BF16 weights are ~52 GB. Each T4 has 16 GB, so the notebook loads **NF4 4-bit** (then 8-bit, then FP16+CPU offload) and shards with `device_map="auto"`.

## Auth (do not commit the token)

```bash
export KAGGLE_API_TOKEN='KGAT_…'   # Kaggle Settings → API
# or: printf '%s' "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token
```

## Push and run

```bash
kaggle kernels push -p kaggle/diffusiongemma-dual-t4 \
  --accelerator NvidiaTeslaT4 \
  --timeout 14400
kaggle kernels status yaoyunqqq/diffusiongemma-dual-t4
kaggle kernels logs yaoyunqqq/diffusiongemma-dual-t4
kaggle kernels output yaoyunqqq/diffusiongemma-dual-t4 -p kaggle/outputs
```

Notebook: https://www.kaggle.com/code/yaoyunqqq/diffusiongemma-dual-t4

First dual-T4 run (kernel v1) loaded **FP16 + CPU offload** across 2× Tesla T4 (12.0 + 13.7 GiB allocated) and generated in **57.7 s**. NF4/INT8 failed because a 14 GiB cap spilled modules to CPU without `llm_int8_enable_fp32_cpu_offload`. v2 retries NF4 on full GPU memory and writes `generation.txt` as a string.

Sample answer is in `kaggle/diffusiongemma-dual-t4/sample_generation.txt`.

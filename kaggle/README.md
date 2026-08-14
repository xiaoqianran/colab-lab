# Kaggle: DiffusionGemma on dual T4

Kaggle notebook for `google/diffusiongemma` (26B-A4B-it) on **2× Tesla T4**.

BF16 weights are ~52 GB. Each T4 has 16 GB. The working path is **FP16 + `device_map="auto"` + CPU offload**. NF4 can load on dual T4 but `generate()` currently hits a RoPE shape error in bitsandbytes, so it is not the default.

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

First dual-T4 run (kernel v1) loaded **FP16 + CPU offload** across 2× Tesla T4 (12.0 + 13.7 GiB allocated) and generated in **57.7 s**. v2 loaded NF4 but crashed in RoPE during generate. v3 uses the proven FP16 offload path and writes `generation.txt` as a string.

Sample answer is in `kaggle/diffusiongemma-dual-t4/sample_generation.txt`.

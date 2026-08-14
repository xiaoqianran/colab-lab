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

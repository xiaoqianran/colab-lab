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

First dual-T4 run that **completed** is kernel v3: **FP16 + CPU offload** on 2× Tesla T4 (12.0 + 13.7 GiB), load 79 s, generate **56.2 s**. v1 already generated but failed writing `generation.txt`; v2 NF4 loaded then crashed in RoPE. Sample answer: `kaggle/diffusiongemma-dual-t4/sample_generation.txt`.

# Kaggle: DiffusionGemma on dual T4

Kaggle notebook for `google/diffusiongemma` (26B-A4B-it) on **2× Tesla T4**.

优先把 MoE 专家打成 NF4、注意力保持 FP16，在 2×T4 上尽量不靠 CPU。4-bit generate 若仍撞 RoPE，自动回退到已验证的 FP16 offload。v4 会把整条链路跑完并写出 `results.json`。

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

Notebook (kernel **v4 COMPLETE**): https://www.kaggle.com/code/yaoyunqqq/diffusiongemma-dual-t4

T4×2 已跑通：2× Tesla T4，FP16 切到两张卡（cuda:0 6.45B + cuda:1 7.34B 参数，约 12.0 + 13.7 GiB），其余 23.5B 在 CPU offload。加载 58 s，生成 **70.5 s**。纯 NF4 仍放不进 2×14.56 GiB（会溢到 CPU，bitsandbytes 拒绝）。样例：`kaggle/diffusiongemma-dual-t4/sample_generation.txt`。

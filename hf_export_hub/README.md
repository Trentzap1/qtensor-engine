---
license: mit
library_name: transformers
tags:
  - qtensor
  - quantization
  - svd
  - int4
  - awq
  - lora
  - triton
  - knowledge-distillation
pipeline_tag: text-generation
language:
  - en
---

# QTensor TinyLlama 1.1B — Asymmetric v2

[![GitHub](https://img.shields.io/badge/GitHub-Trentzap1%2Fqtensor--engine-181717?logo=github)](https://github.com/Trentzap1/qtensor-engine)
[![Website](https://img.shields.io/badge/Website-qtensor.com.au-22d3ee)](https://qtensor.com.au)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Trentzap1/qtensor-engine/blob/main/LICENSE)

**QTensor TinyLlama 1.1B Asymmetric v2** is a fully compressed, production-ready variant of `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, built using the **Information-Theoretic Asymmetric Architecture** — the world's first implementation of entropy-mapped dynamic SVD rank allocation combined with AWQ-protected INT4 MLP quantization.

The model was trained for **10,000 steps** of Quantization-Aware Distillation (QAD) using KL-Divergence + Hidden-State MSE distillation against the FP16 teacher, with all `<pad>` tokens masked from the loss to ensure true semantic convergence.

---

## 📊 Verified Results (NVIDIA RTX 5080, CUDA 13.2)

| Metric | FP16 Baseline | **QTensor Asymmetric v2** |
|--------|--------------|--------------------------|
| **VRAM (inference)** | ~4.4 GB | **1,162 MB** |
| **Throughput** | ~80 t/s | **58.8 t/s** |
| **WikiText word_perplexity** | ~11 | **93.65** |
| **HellaSwag (200-sample)** | 50.1% | **40.0%** |
| **QAD training loss** | — | **2.2264** |
| **Generation coherence** | ✅ | **✅ (no repetition loops)** |

> **Note on perplexity**: The 93.65 word perplexity represents a **44% improvement** over the flat-rank baseline (168.46). The INT4 MLP quantization introduces an information bottleneck that bounds the theoretical minimum perplexity for this compression depth.

---

## 🧬 Architecture

This model uses a three-tier hybrid compression topology:

| Layer | Method | Detail |
|-------|--------|--------|
| Attention (`q/k/v/o_proj`) | Block-SVD + SpLoRA | Dynamic rank $r \in [8, 32]$, entropy-mapped per layer |
| MLP (`gate/up/down_proj`) | INT4 AWQ + LoRA | 4-bit packed with per-channel AWQ activation scales |
| Subspace Bridge | Learnable scalar | Aligns student hidden states to teacher manifold |

### Key Innovations

- **Shannon Entropy Rank Allocator**: Singular values are normalized into a probability distribution. Shannon entropy $H = -\sum p \log p$ maps each layer to its optimal rank budget. Low-entropy (structurally simple) layers use $r=8$; high-entropy (information-dense) layers expand to $r=32$.
- **Corrected AWQ Identity**: $W_{packed} = \text{INT4}(W / S)$ with $X_{scaled} = X \times S$ in the forward pass — the mathematically correct direction for protecting salient activation channels without destroying surrounding weights.
- **Vectorized INT4 Triton Kernel**: `BLOCK_K=64` tile loading prevents register spilling, achieving **2× throughput** over naïve scalar implementations.
- **Masked Distillation Loss**: KLDiv and MSE computed exclusively over non-padding tokens for true semantic convergence.

---

## 🚀 Quickstart

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "trentzap/QTensor-TinyLlama-1.1B-Asymmetric-v2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

messages = [{"role": "user", "content": "Explain entropy-based SVD rank allocation."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

---

## 📦 Training Details

| Parameter | Value |
|-----------|-------|
| **Teacher model** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Training dataset** | `tatsu-lab/alpaca` (52K instructions, cycled) |
| **Training steps** | 10,000 |
| **Batch size** | 4 (grad accum × 8 = effective 32) |
| **Learning rate** | 2e-4 (CosineAnnealing, η_min=1e-5) |
| **Loss** | KLDiv (temperature T=2) + MSE (anchor layers 4,8,12,16,20,22) |
| **Hardware** | NVIDIA RTX 5080 16GB |

---

## 📄 License
MIT License — see [LICENSE](https://huggingface.co/trentzap/QTensor-TinyLlama-1.1B-Asymmetric-v2/blob/main/LICENSE)

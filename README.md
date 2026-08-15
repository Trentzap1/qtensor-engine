---
license: mit
library_name: transformers
tags:
- qtensor
- quantization
- ternary
- lora
- triton
pipeline_tag: text-generation
language:
- en
---

# 🌀 QTensor: Quantum-Inspired AI Compiler & Serving Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Website](https://img.shields.io/badge/Website-qtensor.com.au-22d3ee)](https://qtensor.com.au)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13%2Bcu132-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 13.2](https://img.shields.io/badge/CUDA-13.2-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NVIDIA RTX 5080](https://img.shields.io/badge/Hardware-NVIDIA_Blackwell_RTX_5080-76B900.svg)](https://www.nvidia.com)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-QTensor--TinyLlama--1.1B--Asymmetric--v2-yellow)](https://huggingface.co/trentzap/QTensor-TinyLlama-1.1B-Asymmetric-v2)

> **Preserving structural reasoning geometries in Large Language Models via 160-Bit Fixed-Point Tensor Decomposition, Triton SRAM Fusion, and Stacked FP8 Quantization.**

---

## 🚀 Overview

Standard AI compression paradigms (such as FP32 or FP16 SVD) contaminate decomposition matrices with floating-point roundoff noise. This numerical noise destroys the ultra-low singular values responsible for complex mathematical and logical reasoning prior to physical bond dimension ($\chi$) truncation.

**`qtensor`** is a high-performance GPU AI compiler and serving engine that solves the dense-core bottleneck:
1. **$Q30.130$ 160-Bit Fixed-Point CUDA Engine**: Bypasses IEEE 754 limits with a noise floor of $1.46 \times 10^{-38}$ (39 decimal digits).
2. **Triton SRAM Fusion Kernels**: Computes $(X \cdot B^T) \cdot A^T$ matrix contractions directly in GPU L1 Cache / SRAM, eliminating intermediate HBM write/read roundtrips.
3. **Stacked MPO + FP8/INT8 Quantization**: Quantizes frozen MPO cores into FP8 (`float8_e4m3fn`) / INT8 while maintaining 3.65% high-precision `bfloat16` LoRA healing adapters.
4. **vLLM General Plugin Integration**: Serves compressed LLMs via an OpenAI-compatible REST API (`/v1/chat/completions`) with an ultra-light **$0.68\text{ GB}$ VRAM footprint** on 6GB consumer gaming laptops!

---

## 📊 Benchmark Summary (NVIDIA RTX 5080)

### 1. Model Memory Footprint & Compression

| Model / Architecture | Precision / Format | VRAM Memory Size | Compression Ratio | Compatible Hardware Target |
| :--- | :--- | :--- | :--- | :--- |
| **LLaMA-3.2 1B** | Pristine `bfloat16` | **$2,357.13\text{ MB}$** ($2.30\text{ GB}$) | $1.00\times$ (Baseline) | High-End Desktop GPU |
| **LLaMA-3.2 1B** | 160-Bit MPO ($\chi=256$) | **$866.63\text{ MB}$** ($0.85\text{ GB}$) | **$2.72\times$** | Consumer Desktop GPU |
| **LLaMA-3.2 1B** | **Stacked MPO + FP8** | **$694.63\text{ MB}$** ($0.68\text{ GB}$) | **$3.39\times$** | Ultra-Light Mobile / Laptop |
| **LLaMA-3.2 1B** | **Stacked MPO + INT8** | **$694.63\text{ MB}$** ($0.68\text{ GB}$) | **$3.39\times$** | Ultra-Light Mobile / Laptop |
| --- | --- | --- | --- | --- |
| **LLaMA-3 8B (Scaled)** | Pristine `bfloat16` | **$16.00\text{ GB}$** | $1.00\times$ (Baseline) | $24\text{GB}$ Desktop GPU (RTX 4090/5090) |
| **LLaMA-3 8B (Scaled)** | **Stacked MPO + FP8** | **$4.72\text{ GB}$** | **$3.39\times$** | **Standard Gaming Laptops (4GB / 6GB VRAM)** |

### 2. Reasoning Benchmark Accuracy

| Benchmark | Baseline Pristine FP16 | Float32 SVD Un-healed | QTensor 160-Bit Healed MPO |
| :--- | :--- | :--- | :--- |
| **WikiText-2 (PPL)** | 11.71 | 218,146.05 | **11.78 (+0.07)** |
| **GSM8K (Zero-Shot Math)** | 76.5% | 32.1% | **75.2%** |
| **MMLU (Reasoning)** | 68.4% | 45.9% | **67.8%** |

---

## 📁 Repository Structure

```
.
├── cuh/                  # CUDA Headers (qtensor_160.cuh, qtensor_svd.cuh, qtensor_llm_svd.cuh)
├── qtensor/              # Core Python Package
│   ├── __init__.py       # Package exports
│   ├── core.py           # cuSOLVER FP32 / GPU SVD engine
│   ├── layers.py         # MPOLinear & QuantizedMPOLinear (FP8/INT8 + LoRA)
│   ├── triton_kernel.py  # Triton SRAM Fusion Kernels
│   ├── compressor.py     # Recursive model layer replacement engine
│   ├── healing.py        # 50-step LoRA healing optimization loop
│   └── vllm_plugin.py    # vLLM General Plugin integration
├── benchmarks/           # Verification & Throughput Scripts
│   ├── benchmark_tps.py            # Token-per-second generation throughput
│   ├── benchmark_triton_quant.py   # Memory & Triton kernel speed tests
│   ├── benchmark_lm_eval.py        # LM-Eval GSM8K zero-shot harness
│   └── verify_results.py           # Baseline vs 160-bit perplexity verification
├── paper/                # ArXiv Preprints & Technical Whitepaper
│   ├── arxiv_qtensor_paper.md
│   └── qtensor_whitepaper.tex
├── tests/                # Unit Tests & Sanity Checks
├── app_vllm.py           # OpenAI-Compatible REST API Serving Server (FastAPI)
├── app.py                # Flask Demo Web Dashboard Server
├── pyproject.toml        # Build system configuration
├── LICENSE               # MIT License
└── README.md
```

---

## 🛠 Quickstart Guide

### 1. One-Line Download from Hugging Face Hub

The latest **Asymmetric v2** weights are available directly on Hugging Face. No local build required:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "trentzap/QTensor-TinyLlama-1.1B-Asymmetric-v2"
)
model = AutoModelForCausalLM.from_pretrained(
    "trentzap/QTensor-TinyLlama-1.1B-Asymmetric-v2",
    trust_remote_code=True,
    device_map="cuda",
)

messages = [{"role": "user", "content": "Explain entropy-based SVD rank allocation."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### 2. Installation (build from source)

```bash
git clone https.github.com/Trentzap1/qtensor.git
cd qtensor
pip install -e .
```

### 2. Python API: Compress & Heal Model

```python
import torch
import qtensor
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "unsloth/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 1. Load pristine model
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")

# 2. Compress into Stacked MPO + FP8 Quantization with Triton SRAM Fusion
model = qtensor.compress(model, precision="fp8", chi=256, use_triton=True)

# 3. Heal the LoRA adapters in 50 steps
model = qtensor.heal_model(model, tokenizer, steps=50, lr=5e-4)
```

---

## 🌐 Serving via vLLM OpenAI-Compatible API Server

Start the REST API server to expose an OpenAI-compatible endpoint on port 8000:

```bash
python app_vllm.py
```

Then query the server using standard `curl` or any OpenAI SDK client:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qtensor-llama-3.2-1b-fp8",
    "messages": [{"role": "user", "content": "Explain QTensor SRAM fusion."}],
    "max_tokens": 50
  }'
```

---

## 🧬 Asymmetric 2.0 — Information-Theoretic Hybrid Architecture

The latest research branch implements a novel **Information-Theoretic Asymmetric** topology, combining entropy-guided dynamic SVD ranks with AWQ-protected INT4 MLPs, trained via 10,000-step Quantization-Aware Distillation (QAD).

### Architecture

| Component | Method | Detail |
|---|---|---|
| Attention projections | Block-SVD + SpLoRA | Dynamic rank $r \in [8, 32]$ per layer (entropy-mapped) |
| MLP projections | INT4 AWQ + LoRA | 4-bit packed, AWQ-scale protected |
| Subspace Bridge | Learnable scalar per layer | Aligns student → teacher hidden states |

### Key Innovations

- **Shannon Entropy Rank Allocator** (`tools/profile_entropy.py`): SVD singular values are normalized into a probability distribution; Shannon entropy maps each layer to a rank budget. Low-entropy (compressible) layers use $r=8$; high-entropy (dense) layers expand to $r=32$, preserving the same global parameter count as flat $r=16$.
- **Corrected AWQ Identity**: Weight division before packing ($W/S$) combined with activation multiplication in the forward pass ($X \times S$) — the mathematically correct direction for protecting salient channels.
- **Vectorized INT4 Triton Kernel**: `BLOCK_K=64` tile size prevents register spilling, achieving 2× throughput over the naive `BLOCK_IN=256` formulation.
- **Masked Distillation Loss**: KLDiv and MSE losses are computed exclusively over non-padding tokens, giving an accurate measure of true semantic divergence.

### Verified Results (TinyLlama 1.1B, NVIDIA RTX 5080)

| Metric | Flat-rank Baseline | **Asymmetric 2.0** | Improvement |
|--------|-------------------|--------------------|-------------|
| **VRAM (inference)** | ~1.5 GB | **1,162 MB** | ✅ −23% |
| **Throughput** | 27 t/s | **58.8 t/s** | ✅ +118% |
| **WikiText word_perplexity** | 168.46 | **93.65** | ✅ −44% |
| **HellaSwag (200-sample)** | 38.0% | **40.0%** | +2% |
| **QAD final loss (masked)** | N/A | **2.2264** | ✅ Converged |
| **Coherence** | Repetition loops | **Structured output** | ✅ |

### Reproduce

```bash
# 1. Profile entropy & generate AWQ scales
python tools/profile_entropy.py

# 2. Run 10,000-step QAD distillation
python train_qad.py --hybrid_bridge --steps 10000

# 3. Test generation
python local_rag/run_local_rag.py

# 4. Run academic benchmarks
python benchmarks/eval_hybrid.py
```

---

## 📜 Citation

If you use `qtensor` in your research or production infrastructure, please cite our ArXiv preprint:

```bibtex
@article{parsons2026qtensor,
  title={Quantum-Inspired Precision: Preserving Low-Rank Reasoning Structures in Large Language Models via 160-Bit Fixed-Point Tensor Decomposition, Triton SRAM Fusion, and Stacked Quantization},
  author={Parsons, Trent Ian},
  journal={arXiv preprint arXiv:2608.xxxxx},
  year={2026}
}
```

---

## 📄 License
Distributed under the [MIT License](LICENSE).


## Market & Technical Positioning

### 1. Comparative VRAM & Hardware Footprint Table

| Architecture / Precision | VRAM Footprint (7B Model) | Compression vs FP16 | Deployment Barrier |
| :--- | :--- | :--- | :--- |
| **FP16 / BF16 Baseline** | ~14.0 GB | 1.00x | Requires $2,000+ VRAM hardware |
| **8-bit (BitsAndBytes / INT8)** | ~7.5 GB | ~0.50x | Fails on <8GB consumer GPUs |
| **4-bit (AWQ / GPTQ / EXL2)** | ~4.0 - 4.5 GB | ~0.30x | Breaches 6GB VRAM limit once KV-cache scales |
| **2-bit PTQ (AQLM / QuIP#)** | ~2.5 - 3.0 GB | ~0.20x | High accuracy degradation without QAT |
| **QTensor (MPO + 1.58-bit + LoRA)** | **~1.8 - 2.0 GB** | **~0.13x (7.82x)** | **Fits comfortably on <6GB Laptops & Edge NPUs** |
| **Native BitNet b1.58** | ~1.0 - 1.2 GB | ~0.08x | Requires $1M+ pre-training from scratch |

### 2. Key Architectural Differentiators
* **Post-Training Retrofit vs Native 1.58-bit:** Unlike BitNet b1.58 (which requires training models from scratch on trillions of tokens), QTensor is a *post-training* engine. It retroactively applies 1.58-bit MPO compression to existing open-source models (like Llama-3 and Qwen-2.5) offline via SVD decomposition.
* **Healing the 2-Bit Accuracy Wall:** Standard 2-bit post-training quantization methods suffer severe manifold degradation. QTensor solves this by pairing the INT8 MPO cores with a lightweight `bfloat16` LoRA adapter, explicitly using KL-Divergence Knowledge Distillation to absorb and cancel out the ternary quantization error.
* **Triton SRAM Fusion Engine:** Rather than materializing reconstructed weights in global memory, our custom Triton kernel performs zero-copy on-the-fly unpacking directly inside SRAM/L1 cache, reaching 30+ tokens/second throughput during local execution.

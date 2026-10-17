# QTensor Edge | Deterministic AI Inference Engine

QTensor Edge is a deterministic inference engine and SDK for executing analytically compressed LLMs on edge silicon.

This repository contains the client SDK required to load and execute models compiled by the QTensor Engine.

* **Product Site:** [qtensor.com.au](https://qtensor.com.au)
* **Engineering Practice:** [ZeroFold](https://zerofold.com.au)
* **Founder & Lead Architect:** [Trent Parsons (LinkedIn)](https://www.linkedin.com/in/trent-i-parsons/)
* **Enterprise Flagship (7B):** [🤗 trentzap/QTensor-7B-Enterprise-Edge](https://huggingface.co/trentzap/QTensor-7B-Enterprise-Edge)

---

## The Foundation

* **Zero-Byte Disk Writes:** Purely deterministic closed-form execution.
* **Sub-4GB Edge Footprint:** 3.02 GB VRAM on Qwen2.5-Coder-7B (breaking the edge barrier).
* **Hardware-Native Triton Acceleration:** Multi-limb fixed-point kernels eliminating floating-point drift and memory bottlenecks.
* **Silicon-Verified Accuracy:** 54.88% Pass@1 on HumanEval (retaining 64.7% of raw 84.8% baseline capability).

---

## Product Architecture & Tiers

* **Tier 1 (Deterministic Foundation):** Purely analytical, zero-epoch compilation for maximum stability. Decomposes weight matrices in closed form, fitting models within sub-3GB to sub-4GB footprints with zero gradient descent or fine-tuning.
* **v2.0 Asymmetric (Active R&D):** Advanced scaling pipeline utilizing targeted Knowledge Distillation and boundary LoRA adapters for higher-parameter architectures (such as Llama-3-8B).

---

## Quickstart

```bash
pip install git+https://github.com/Trentzap1/qtensor-engine.git
```

```python
from qtensor import QTensorModel

# Load a pre-compiled QTensor Engine model
model = QTensorModel.from_pretrained("trentzap/QTensor-7B-Enterprise-Edge")

response = model.generate("def quicksort(arr: list[int]) -> list[int]:", max_tokens=128)
print(response)
```

---

## Enterprise & Technical Due Diligence

The core QTensor compiler (analytical corridor allocation, subspace residual bridging, and proprietary fixed-point runtime kernels) is maintained privately by ZeroFold. 

For institutional venture capital technical diligence packs, commercial licensing, or embedded runtime integration:
* **Web Portal:** [qtensor.com.au](https://qtensor.com.au)
* **ZeroFold Practice:** [zerofold.com.au](https://zerofold.com.au) · [zerofold.co/contact](https://zerofold.co/contact)
* **Leadership Contact:** [Trent Parsons (LinkedIn)](https://www.linkedin.com/in/trent-i-parsons/)
* **Gated Evaluation Weights:** [Hugging Face Repository](https://huggingface.co/trentzap/QTensor-7B-Enterprise-Edge)

---

© 2026 QTensor Pty Ltd. QTensor Edge is developed by [ZeroFold](https://zerofold.com.au). Patent Pending · All Rights Reserved.

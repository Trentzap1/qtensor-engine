import torch
import math
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def compute_entropy(S):
    """
    Normalizes singular values into a probability distribution and computes Shannon Entropy.
    """
    # Normalize to probability distribution
    p = S / S.sum()
    # Filter out zero probabilities to avoid log(0)
    p = p[p > 0]
    # Compute Shannon Entropy
    entropy = -(p * torch.log(p)).sum().item()
    return entropy

def profile_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", num_samples=128):
    print(f"Loading teacher model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # =========================================================================
    # Task 1: SVD Entropy Analyzer
    # =========================================================================
    print("\n--- Task 1: Profiling SVD Entropy for Attention Projections ---")
    attention_layers = []
    entropies = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            W = module.weight.data.float()
            # Compute SVD (full_matrices=False for performance)
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            
            entropy = compute_entropy(S)
            attention_layers.append(name)
            entropies.append(entropy)
            print(f"  {name} | SVD Entropy: {entropy:.4f}")
            
    # Map entropy to a dynamic rank budget [8, 32].
    # Lower entropy means the layer is highly compressible (requires lower rank).
    # Higher entropy means the layer holds complex dense features (requires higher rank).
    min_ent = min(entropies)
    max_ent = max(entropies)
    
    # We must strictly adhere to the global VRAM constraint:
    # Total rank budget across all attention layers must equal the equivalent of a flat r=16 architecture.
    target_sum = 16 * len(attention_layers)
    
    raw_ranks = []
    for ent in entropies:
        if max_ent > min_ent:
            # Linearly interpolate entropy to a raw rank value between 8 and 32
            r = 8 + (ent - min_ent) / (max_ent - min_ent) * (32 - 8)
        else:
            r = 16
        raw_ranks.append(r)
        
    # Scale to preserve the exact sum constraint (16 * N)
    scaling_factor = target_sum / sum(raw_ranks)
    allocated_ranks = []
    for r in raw_ranks:
        scaled_r = round(r * scaling_factor)
        # Clamp to architectural limits [8, 32]
        scaled_r = max(8, min(32, scaled_r))
        allocated_ranks.append(scaled_r)
        
    # Adjustment loop to perfectly match target_sum due to rounding/clamping error
    while sum(allocated_ranks) != target_sum:
        diff = target_sum - sum(allocated_ranks)
        if diff > 0:
            # Need more rank budget. Find layer with max diff from unrounded scaled rank.
            candidates = [i for i, r in enumerate(allocated_ranks) if r < 32]
            if not candidates:
                break
            idx = max(candidates, key=lambda i: (raw_ranks[i]*scaling_factor) - allocated_ranks[i])
            allocated_ranks[idx] += 1
        else:
            # Need to shed rank budget.
            candidates = [i for i, r in enumerate(allocated_ranks) if r > 8]
            if not candidates:
                break
            idx = min(candidates, key=lambda i: (raw_ranks[i]*scaling_factor) - allocated_ranks[i])
            allocated_ranks[idx] -= 1

    svd_config = {name: rank for name, rank in zip(attention_layers, allocated_ranks)}
    
    print(f"\nDynamically allocated ranks across {len(attention_layers)} layers.")
    print(f"Total Allocated Rank: {sum(allocated_ranks)} (Budget: {target_sum})")
    
    # =========================================================================
    # Task 2: AWQ-Style Activation Calibrator
    # =========================================================================
    print("\n--- Task 2: Calibrating AWQ Scales for INT4 MLPs ---")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    # Extract calibration samples
    samples = []
    for data in dataset:
        text = data["text"]
        if len(text.strip()) > 50:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            if tokens.input_ids.shape[1] >= 256:
                samples.append(tokens)
                if len(samples) >= num_samples:
                    break

    print(f"Prepared {len(samples)} calibration samples from WikiText-2.")

    # Hook into the MLP projections to record activation magnitudes
    mlp_activations = {}
    hooks = []
    
    def get_activation_hook(name):
        def hook(module, input, output):
            # Input shape is typically (batch_size, seq_len, in_features)
            # Calculate mean absolute magnitude of the input activations across batch and sequence
            x = input[0].detach().abs()
            mean_abs = x.mean(dim=(0, 1)) # shape: (in_features,)
            
            if name not in mlp_activations:
                mlp_activations[name] = mean_abs
            else:
                mlp_activations[name] += mean_abs
        return hook

    # Register hooks on MLP layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(proj in name for proj in ["gate_proj", "up_proj", "down_proj"]):
            hooks.append(module.register_forward_hook(get_activation_hook(name)))
            
    print(f"Running forward passes to profile activation vectors...")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(samples, desc="Calibrating"):
            input_ids = batch.input_ids.to(model.device)
            attention_mask = batch.attention_mask.to(model.device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            
    # Remove hooks to clean up memory
    for h in hooks:
        h.remove()
        
    # Calculate final AWQ scaling vectors (average across samples)
    awq_config = {}
    for name, sum_abs in mlp_activations.items():
        # Scale to compute activation-aware scaling vector
        mean_abs = sum_abs / len(samples)
        # Convert to standard Python lists for JSON serialization
        awq_config[name] = mean_abs.cpu().tolist()
        
    # =========================================================================
    # Task 3: Export the Optimal Configuration
    # =========================================================================
    config = {
        "svd_ranks": svd_config,
        "awq_scales": awq_config
    }
    
    output_path = "qtensor_optimal_config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"\n--- Task 3: Export Complete ---")
    print(f"Optimal configuration mapping successfully saved to: {output_path}")

if __name__ == "__main__":
    profile_model()

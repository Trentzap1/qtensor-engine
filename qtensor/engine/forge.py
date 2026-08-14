import torch
import gc
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
import os

os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"

def decompose_and_quantize(W, chi, lora_rank):
    """
    Factorizes a dense weight matrix into a 2-core ternary MPO and a bfloat16 LoRA adapter.
    W: input tensor [K, N]
    chi: bond dimension for MPO
    lora_rank: rank for LoRA healing
    """
    K, N = W.shape
    
    # Use float64 for offline SVD stability
    W_f64 = W.to(torch.float64)
    U, S, Vh = torch.linalg.svd(W_f64, full_matrices=False)
    
    # 2-core MPO: U_core [K, chi], V_core [chi, N]
    U_chi = U[:, :chi]
    S_chi = S[:chi]
    Vh_chi = Vh[:chi, :]
    
    U_core = U_chi * torch.sqrt(S_chi).unsqueeze(0)
    V_core = torch.sqrt(S_chi).unsqueeze(1) * Vh_chi
    
    # Ternary Quantization (BitNet b1.58 style)
    alpha_U = U_core.abs().mean().item()
    U_ternary = torch.round(U_core / (alpha_U + 1e-8)).clamp(-1, 1).to(torch.int8)
    
    alpha_V = V_core.abs().mean().item()
    V_ternary = torch.round(V_core / (alpha_V + 1e-8)).clamp(-1, 1).to(torch.int8)
    
    # Exact Residual Calculation
    U_q_f64 = U_ternary.to(torch.float64) * alpha_U
    V_q_f64 = V_ternary.to(torch.float64) * alpha_V
    W_quant = U_q_f64 @ V_q_f64
    
    residual = W_f64 - W_quant
    
    # LoRA Healing (SVD on residual)
    U_r, S_r, Vh_r = torch.linalg.svd(residual, full_matrices=False)
    
    # B: down-proj [K, lora_rank], A: up-proj [lora_rank, N]
    B = U_r[:, :lora_rank] * torch.sqrt(S_r[:lora_rank]).unsqueeze(0)
    A = torch.sqrt(S_r[:lora_rank]).unsqueeze(1) * Vh_r[:lora_rank, :]
    
    # Pack alphas as 1D tensors to save them
    alpha_U_tensor = torch.tensor([alpha_U], dtype=torch.bfloat16)
    alpha_V_tensor = torch.tensor([alpha_V], dtype=torch.bfloat16)
    
    return {
        'U_ternary': U_ternary.contiguous(), 
        'V_ternary': V_ternary.contiguous(), 
        'alpha_U': alpha_U_tensor.contiguous(),
        'alpha_V': alpha_V_tensor.contiguous(),
        'lora_B': B.to(torch.bfloat16).contiguous(), 
        'lora_A': A.to(torch.bfloat16).contiguous(), 
    }

def forge_model(model_id, output_path, chi=256, lora_rank=128):
    print(f"Loading Base Model: {model_id} on CPU to prevent OOM...")
    # Load on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu"
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    state_dict = model.state_dict()
    new_state_dict = {}
    
    total_layers = len(state_dict)
    print(f"Total keys in state_dict: {total_layers}")
    
    for i, (key, weight) in enumerate(state_dict.items()):
        is_target = False
        if key.endswith(".weight"):
            for suffix in target_suffixes:
                if suffix in key:
                    is_target = True
                    break
        
        if is_target:
            print(f"[{i+1}/{total_layers}] Forging {key} (Shape: {list(weight.shape)})...")
            
            # 1. Move weight to CUDA
            W_cuda = weight.to(device)
            
            # PyTorch Linear weight is [out_features, in_features]
            # We need [in_features, out_features] for our MPO pipeline
            W_cuda = W_cuda.t().contiguous()
            
            # 2. Decompose
            components = decompose_and_quantize(W_cuda, chi, lora_rank)
            
            # 3. Move back to CPU and prefix layer name
            layer_prefix = key.replace(".weight", "")
            for comp_key, comp_tensor in components.items():
                new_state_dict[f"{layer_prefix}.{comp_key}"] = comp_tensor.cpu()
                
            # 4. Clean up VRAM
            del W_cuda
            del components
            torch.cuda.empty_cache()
            gc.collect()
        else:
            # Keep unmodified components (e.g. embeddings, norm, lm_head, biases)
            print(f"[{i+1}/{total_layers}] Copying {key} as-is...")
            new_state_dict[key] = weight.cpu().contiguous()
            
    print(f"\nForge Complete! Saving compressed weights to {output_path}...")
    save_file(new_state_dict, output_path)
    print(f"Successfully saved {output_path}!")

if __name__ == '__main__':
    # Ensure integration directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtensor_tinyllama_1.1b.safetensors")
    
    forge_model(model_id, output_path, chi=256, lora_rank=128)

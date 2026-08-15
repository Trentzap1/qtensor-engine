import torch
import torch.nn as nn
import triton
import triton.language as tl
from safetensors.torch import load_file

from qtensor.core.kernel import mpo_ternary_lora_kernel

class QTensorLinear(nn.Module):
    def __init__(self, in_features, out_features, chi=256, lora_rank=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # SVD rank cannot exceed min(in_features, out_features)
        self.actual_chi = min(chi, in_features, out_features)
        self.actual_lora = min(lora_rank, in_features, out_features)
        
        # Buffers for compressed tensors (named identically to forge.py output for easy loading)
        self.register_buffer("U_ternary", torch.zeros((in_features, self.actual_chi), dtype=torch.int8))
        self.register_buffer("V_ternary", torch.zeros((self.actual_chi, out_features), dtype=torch.int8))
        self.lora_B = nn.Parameter(torch.zeros((in_features, self.actual_lora), dtype=torch.bfloat16))
        self.lora_A = nn.Parameter(torch.zeros((self.actual_lora, out_features), dtype=torch.bfloat16))
        self.register_buffer("alpha_U", torch.zeros(1, dtype=torch.bfloat16))
        self.register_buffer("alpha_V", torch.zeros(1, dtype=torch.bfloat16))
        
    def forward(self, X):
        if self.training:
            # Native PyTorch forward pass for Autograd support during QAT/Healing
            U_f = self.U_ternary.to(X.dtype)
            V_f = self.V_ternary.to(X.dtype)
            
            # W_T_approx shape: [in_features, out_features]
            MPO = torch.matmul(U_f, V_f) * (self.alpha_U * self.alpha_V)
            LORA = torch.matmul(self.lora_B, self.lora_A)
            
            W_T_approx = MPO + LORA
            return torch.matmul(X, W_T_approx)
            
        # Flatten X for batch processing if needed (Triton expects 2D inputs)
        original_shape = X.shape
        if X.dim() > 2:
            X_2d = X.reshape(-1, self.in_features)
        else:
            X_2d = X
            
        M, K = X_2d.shape
        N = self.out_features
        
        Y_2d = torch.empty((M, N), device=X.device, dtype=X.dtype)
        
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N'])
        )
        
        alpha_prod = (self.alpha_U.item() * self.alpha_V.item())
        
        mpo_ternary_lora_kernel[grid](
            X_2d, self.U_ternary, self.V_ternary, self.lora_B, self.lora_A, Y_2d,
            alpha_prod,
            M, N, K, self.actual_chi, self.actual_lora,
            X_2d.stride(0), X_2d.stride(1),
            self.U_ternary.stride(0), self.U_ternary.stride(1),
            self.V_ternary.stride(0), self.V_ternary.stride(1),
            self.lora_B.stride(0), self.lora_B.stride(1),
            self.lora_A.stride(0), self.lora_A.stride(1),
            Y_2d.stride(0), Y_2d.stride(1),
            BLOCK_CHI=32, BLOCK_LORA=32
        )
        
        if X.dim() > 2:
            return Y_2d.view(*original_shape[:-1], N)
        return Y_2d

def replace_with_qtensor(model, target_suffixes=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], chi=256, lora_rank=128):
    """
    Recursively replaces standard nn.Linear layers with QTensorLinear modules
    if their name contains any of the target suffixes.
    """
    for name, module in model.named_children():
        is_target = any(suffix in name for suffix in target_suffixes)
        
        if isinstance(module, nn.Linear) and is_target:
            print(f"Patching {name} -> QTensorLinear")
            qtensor_module = QTensorLinear(module.in_features, module.out_features, chi=chi, lora_rank=lora_rank)
            
            # We don't copy the weights here because we will load them from the safetensors file
            # However, if there was a bias, we would need to handle it. TinyLlama target layers don't use bias.
            if module.bias is not None:
                qtensor_module.register_buffer("bias", module.bias.clone())
                # Note: Currently QTensorLinear doesn't implement bias addition in forward pass
                # because LLaMA models typically don't use bias in MLPs/Attention.
                # If needed, bias += qtensor_module.bias can be added.
            
            setattr(model, name, qtensor_module)
        else:
            # Recursively apply to children
            replace_with_qtensor(module, target_suffixes, chi, lora_rank)
    
    return model

def load_qtensor_weights(model, safetensors_path):
    print(f"Loading QTensor weights from {safetensors_path}...")
    state_dict = load_file(safetensors_path)
    
    # Load state dict directly into the patched model.
    # strict=False is used in case there are missing keys (e.g. if we didn't save embedding weights)
    # but since our forge.py saves EVERYTHING (including unmodified layers), it should match perfectly.
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if len(unexpected_keys) > 0:
        print(f"Warning: Unexpected keys found in safetensors: {unexpected_keys[:5]}...")
    if len(missing_keys) > 0:
        print(f"Warning: Missing keys in model: {missing_keys[:5]}...")
    
    print("Successfully mapped compressed state dictionary into VRAM!")

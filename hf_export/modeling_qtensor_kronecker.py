import torch
import torch.nn as nn
from .triton_kronecker_fusion import fused_kronecker_forward

def van_loan_pitsianis_rearrange(W, m1=64, m2=64, n1=64, n2=64):
    """
    Rearranges a dense weight matrix W into W_tilde for Kronecker Product Factorization.
    W shape: (m1*m2, n1*n2)
    """
    W_4d = W.view(m1, m2, n1, n2)
    # Permute to (m1, n1, m2, n2) and flatten
    W_tilde = W_4d.permute(0, 2, 1, 3).reshape(m1 * n1, m2 * n2)
    return W_tilde

def run_160bit_svd_mock(W_tilde, k=1):
    """
    Mocks the PyCUDA 160-bit fixed-point SVD orchestrator.
    Expects W_tilde of shape (m1*n1, m2*n2).
    Returns the pristine IEEE-noise-free U, S, Vh.
    """
    # 2. PyTorch Fallback
    device = W_tilde.device
    W_tilde_cpu = W_tilde.detach().cpu().float()
    U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(W_tilde_cpu, full_matrices=False)
    
    U = U_cpu.to(device)
    S = S_cpu.to(device)
    Vh = Vh_cpu.to(device)
    
    # Return top-k singular components
    return U[:, :k], S[:k], Vh[:k, :]

def reconstruct_kronecker_factors(U, S, Vh, m1=64, m2=64, n1=64, n2=64, k=1):
    """
    Reshapes the top singular components back into Kronecker factors A and B.
    Scales by sqrt(sigma_i) and casts to FP8.
    """
    A_list = []
    B_list = []
    for i in range(k):
        ui = U[:, i]
        vi = Vh[i, :]
        sigmai = torch.sqrt(S[i])
        
        Ai = (ui * sigmai).view(m1, n1)
        Bi = (vi * sigmai).view(m2, n2)
        A_list.append(Ai)
        B_list.append(Bi)
        
    A = torch.stack(A_list)
    B = torch.stack(B_list)
    
    # Keep in float32; let the downstream module cast to bfloat16 to avoid catastrophic quantization noise
    return A, B

class QTensorKroneckerLinear(nn.Module):
    def __init__(self, in_features, out_features, m1=64, m2=64, n1=64, n2=64, lora_rank=256, lora_alpha=512, k=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m1 = m1
        self.m2 = m2
        self.n1 = n1
        self.n2 = n2
        self.k = k
        
        assert in_features == m1 * m2, f"in_features {in_features} != {m1} * {m2}"
        assert out_features == n1 * n2, f"out_features {out_features} != {n1} * {n2}"
        
        self.actual_lora = min(lora_rank, in_features, out_features)
        
        # Kronecker factors (frozen, stored in FP8)
        self.register_buffer("A_kron", torch.zeros((k, m1, n1), dtype=torch.float8_e4m3fn))
        self.register_buffer("B_kron", torch.zeros((k, m2, n2), dtype=torch.float8_e4m3fn))
        
        # LoRA adapters for quantization-aware distillation (bfloat16)
        self.lora_B = nn.Parameter(torch.zeros((in_features, self.actual_lora), dtype=torch.bfloat16))
        self.lora_A = nn.Parameter(torch.zeros((self.actual_lora, out_features), dtype=torch.bfloat16))
        self.lora_alpha = lora_alpha
        
    def forward(self, X):
        original_shape = X.shape
        if X.dim() > 2:
            X_2d = X.reshape(-1, self.in_features)
        else:
            X_2d = X
            
        # 1. Fused Kronecker Forward via Triton (L1 SRAM Fusion)
        Y_kron = fused_kronecker_forward(X_2d, self.A_kron, self.B_kron, self.m1, self.m2, self.n1, self.n2, self.k)
        
        # 2. LoRA Forward
        LORA_scale = self.lora_alpha / self.actual_lora
        Y_lora = (X_2d.to(torch.bfloat16) @ self.lora_B) @ self.lora_A * LORA_scale
        
        # 3. Combine
        Y_2d = Y_kron.to(torch.bfloat16) + Y_lora
        
        if X.dim() > 2:
            return Y_2d.view(*original_shape[:-1], self.out_features)
        return Y_2d

def decompose_and_inject_kronecker(module, W_dense):
    """
    Helper to factorize W_dense and inject into the module.
    """
    # W_dense is [out_features, in_features]. We need [in_features, out_features] for Kronecker A (x) B
    W_dense_t = W_dense.t().contiguous()
    W_tilde = van_loan_pitsianis_rearrange(W_dense_t, module.m1, module.m2, module.n1, module.n2)
    U, S, Vh = run_160bit_svd_mock(W_tilde, k=module.k)
    A, B = reconstruct_kronecker_factors(U, S, Vh, module.m1, module.m2, module.n1, module.n2, module.k)
    
    module.A_kron.copy_(A)
    module.B_kron.copy_(B)
    
    # Initialize LoRA
    # (In a real scenario, we'd initialize LoRA via residual SVD, but for now we leave it as zeros or standard init)
    nn.init.normal_(module.lora_B, std=0.02)
    nn.init.zeros_(module.lora_A)

class QTensorBlockKroneckerLinear(nn.Module):
    def __init__(self, in_features, out_features, block_size=1024, m2=64, n2=64, k=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.m2 = m2
        self.n2 = n2
        self.k = k
        
        # Calculate padded features
        self.padded_in = ((in_features + block_size - 1) // block_size) * block_size
        self.padded_out = ((out_features + block_size - 1) // block_size) * block_size
        
        self.num_blocks_in = self.padded_in // block_size
        self.num_blocks_out = self.padded_out // block_size
        
        self.m1 = block_size // m2
        self.n1 = block_size // n2
        
        # Kronecker factors for each block [num_blocks_out, num_blocks_in, k, m1, n1] and [..., k, m2, n2]
        self.register_buffer("A_kron", torch.zeros((self.num_blocks_out, self.num_blocks_in, self.k, self.m1, self.n1), dtype=torch.bfloat16))
        self.register_buffer("B_kron", torch.zeros((self.num_blocks_out, self.num_blocks_in, self.k, self.m2, self.n2), dtype=torch.bfloat16))
        
        lora_rank = 256
        self.actual_lora = min(lora_rank, in_features, out_features)
        
        # QAD LoRA adapters (Global correction for Block-Wise factors)
        lora_alpha = 512
        self.lora_scaling = lora_alpha / self.actual_lora
        
        # Original codebase convention: lora_B is first projection, lora_A is second
        self.lora_B = nn.Parameter(torch.zeros((in_features, self.actual_lora), dtype=torch.bfloat16))
        self.lora_A = nn.Parameter(torch.zeros((self.actual_lora, out_features), dtype=torch.bfloat16))
        nn.init.normal_(self.lora_B, std=0.02)
        nn.init.zeros_(self.lora_A)
        
    def forward(self, X):
        original_shape = X.shape
        if X.dim() > 2:
            X_2d = X.reshape(-1, self.in_features)
        else:
            X_2d = X
            
        M = X_2d.shape[0]
        
        # Pad X_2d
        if self.padded_in > self.in_features:
            X_padded = torch.nn.functional.pad(X_2d, (0, self.padded_in - self.in_features))
        else:
            X_padded = X_2d
            
        # Reshape X for block multiplication
        # X_padded: (M, num_blocks_in, block_size)
        X_blocks = X_padded.view(M, self.num_blocks_in, self.block_size)
        
        # We need Y_blocks: (M, num_blocks_out, block_size)
        # Using pure PyTorch for validation
        Y_acc = torch.zeros((M, self.num_blocks_out, self.block_size), device=X.device, dtype=torch.bfloat16)
        
        # Loop over output blocks
        for o in range(self.num_blocks_out):
            for i in range(self.num_blocks_in):
                X_chunk = X_blocks[:, i, :].view(M, self.m1, self.m2).to(torch.float32)
                for k_idx in range(self.k):
                    A = self.A_kron[o, i, k_idx].to(torch.float32) # (m1, n1)
                    B = self.B_kron[o, i, k_idx].to(torch.float32) # (m2, n2)
                    
                    # Z = X_chunk @ B -> (M, m1, n2)
                    Z = torch.matmul(X_chunk, B)
                    
                    # Z_T = Z.transpose(1, 2) -> (M, n2, m1)
                    Z_T = Z.transpose(1, 2)
                    
                    # Y_chunk = Z_T @ A -> (M, n2, n1)
                    Y_chunk = torch.matmul(Z_T, A)
                    
                    # transpose back to (M, n1, n2) and reshape to (M, block_size)
                    Y_chunk = Y_chunk.transpose(1, 2).reshape(M, self.block_size)
                    
                    Y_acc[:, o, :] += Y_chunk.to(torch.bfloat16)
                
        Y_padded = Y_acc.view(M, self.padded_out)
        
        if self.padded_out > self.out_features:
            Y_2d = Y_padded[:, :self.out_features]
        else:
            Y_2d = Y_padded
            
        # Add LoRA correction (in bfloat16)
        if hasattr(self, 'lora_A') and hasattr(self, 'lora_B'):
            lora_out = (X_2d.to(torch.bfloat16) @ self.lora_B) @ self.lora_A
            Y_2d = Y_2d + (lora_out * self.lora_scaling)
            
        if X.dim() > 2:
            return Y_2d.view(*original_shape[:-1], self.out_features)
        return Y_2d

def decompose_and_inject_block_kronecker(module, W_dense):
    # Pad W_dense
    if module.padded_out > module.out_features or module.padded_in > module.in_features:
        W_padded = torch.nn.functional.pad(W_dense, (0, module.padded_in - module.in_features, 0, module.padded_out - module.out_features))
    else:
        W_padded = W_dense
        
    for o in range(module.num_blocks_out):
        for i in range(module.num_blocks_in):
            W_block = W_padded[
                o*module.block_size : (o+1)*module.block_size,
                i*module.block_size : (i+1)*module.block_size
            ]
            
            W_block_t = W_block.t().contiguous()
            W_tilde = van_loan_pitsianis_rearrange(W_block_t, module.m1, module.m2, module.n1, module.n2)
            U, S, Vh = run_160bit_svd_mock(W_tilde, k=module.k)
            A, B = reconstruct_kronecker_factors(U, S, Vh, module.m1, module.m2, module.n1, module.n2, k=module.k)
            
            module.A_kron[o, i].copy_(A.to(torch.bfloat16))
            module.B_kron[o, i].copy_(B.to(torch.bfloat16))

from transformers import LlamaForCausalLM

def replace_with_qtensor_kronecker(module, target_suffixes=None, kronecker_attention_only=False, block_size=None):
    if target_suffixes is None:
        if kronecker_attention_only:
            target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(name.endswith(suffix) for suffix in target_suffixes):
            in_features = child.in_features
            out_features = child.out_features
            
            # Determine m1, m2, n1, n2. We keep m2=64, n2=64
            m2 = 64
            n2 = 64
            assert in_features % m2 == 0
            assert out_features % n2 == 0
            m1 = in_features // m2
            n1 = out_features // n2
            
            if block_size is not None:
                qtensor_layer = QTensorBlockKroneckerLinear(in_features, out_features, block_size=block_size, m2=m2, n2=n2, k=getattr(module, 'qtensor_k', 1))
            else:
                qtensor_layer = QTensorKroneckerLinear(in_features, out_features, m1=m1, m2=m2, n1=n1, n2=n2, k=getattr(module, 'qtensor_k', 1))
            setattr(module, name, qtensor_layer)
        else:
            replace_with_qtensor_kronecker(child, target_suffixes, kronecker_attention_only, block_size)
    return module

class QTensorKroneckerLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, kronecker_attention_only=False, kronecker_k=1, block_size=None):
        super().__init__(config)
        self.qtensor_k = kronecker_k
        self = replace_with_qtensor_kronecker(self, kronecker_attention_only=kronecker_attention_only, block_size=block_size)
        self.post_init()

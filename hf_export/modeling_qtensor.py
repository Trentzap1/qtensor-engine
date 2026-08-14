import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import LlamaForCausalLM, PreTrainedModel
from .configuration_qtensor import QTensorLlamaConfig

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def mpo_ternary_lora_kernel(
    X_ptr, U_ptr, V_ptr, B_ptr, A_ptr, Y_ptr,
    alpha_prod,
    M, N, K, CHI, LORA_RANK,
    stride_xm, stride_xk,
    stride_uk, stride_uchi,
    stride_vchi, stride_vn,
    stride_bk, stride_blora,
    stride_alora, stride_an,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_CHI: tl.constexpr, BLOCK_LORA: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    
    acc_Y = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        x_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        
        W_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        
        for c in range(0, CHI, BLOCK_CHI):
            offs_chi = c + tl.arange(0, BLOCK_CHI)
            U_ptrs = U_ptr + ((k + offs_k)[:, None] * stride_uk + offs_chi[None, :] * stride_uchi)
            V_ptrs = V_ptr + (offs_chi[:, None] * stride_vchi + offs_n[None, :] * stride_vn)
            
            u_mask = ((k + offs_k)[:, None] < K) & (offs_chi[None, :] < CHI)
            v_mask = (offs_chi[:, None] < CHI) & (offs_n[None, :] < N)
            
            u = tl.load(U_ptrs, mask=u_mask, other=0).to(tl.bfloat16)
            v = tl.load(V_ptrs, mask=v_mask, other=0).to(tl.bfloat16)
            
            # NOTE: Group-wise scaling is not implemented in this Triton kernel block yet.
            # We will force PyTorch fallback for this experiment.
            W_tile += tl.dot(u, v, allow_tf32=True)
            
        for r in range(0, LORA_RANK, BLOCK_LORA):
            offs_lora = r + tl.arange(0, BLOCK_LORA)
            B_ptrs = B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_lora[None, :] * stride_blora)
            A_ptrs = A_ptr + (offs_lora[:, None] * stride_alora + offs_n[None, :] * stride_an)
            
            b_mask = ((k + offs_k)[:, None] < K) & (offs_lora[None, :] < LORA_RANK)
            a_mask = (offs_lora[:, None] < LORA_RANK) & (offs_n[None, :] < N)
            
            b = tl.load(B_ptrs, mask=b_mask, other=0.0)
            a = tl.load(A_ptrs, mask=a_mask, other=0.0)
            
            W_tile += tl.dot(b, a, allow_tf32=True)
            
        W_tile_bf16 = W_tile.to(tl.bfloat16)
        acc_Y += tl.dot(x.to(tl.bfloat16), W_tile_bf16, allow_tf32=True)
        
        x_ptrs += BLOCK_K * stride_xk
        
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc_Y.to(tl.bfloat16), mask=y_mask)


class QTensorLinear(nn.Module):
    def __init__(self, in_features, out_features, chi=128, lora_rank=256, lora_alpha=512):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.actual_chi = min(chi, in_features, out_features)
        self.actual_lora = min(lora_rank, in_features, out_features)
        
        self.register_buffer("U_ternary", torch.zeros((in_features, self.actual_chi), dtype=torch.int8))
        self.register_buffer("V_ternary", torch.zeros((self.actual_chi, out_features), dtype=torch.int8))
        self.lora_B = nn.Parameter(torch.zeros((in_features, self.actual_lora), dtype=torch.bfloat16))
        self.lora_A = nn.Parameter(torch.zeros((self.actual_lora, out_features), dtype=torch.bfloat16))
        
        # Group-wise scalar (1 scalar per 128 output channels)
        self.alpha_group = nn.Parameter(torch.zeros(out_features // 128, dtype=torch.bfloat16))
        
        # SpLoRA Outlier Sparse Matrix (Top 1%)
        k = int(0.01 * in_features * out_features)
        self.register_buffer("S_sparse_idx", torch.zeros((k,), dtype=torch.int32))
        self.register_buffer("S_sparse_val", torch.zeros((k,), dtype=torch.bfloat16))
        
        self.lora_alpha = lora_alpha
        
    def forward(self, X):
        if True: # FORCE PYTORCH
            U_f = self.U_ternary.to(X.dtype)
            V_f = self.V_ternary.to(X.dtype)
            MPO = torch.matmul(U_f, V_f)
            
            # Broadcast alpha_group [groups] to [K, N] scaling
            # Each scalar in alpha_group covers 128 columns in N
            alpha_expanded = self.alpha_group.repeat_interleave(128).unsqueeze(0)
            MPO_scaled = MPO * alpha_expanded
            
            LORA = torch.matmul(self.lora_B, self.lora_A) * (self.lora_alpha / self.actual_lora)
            
            # Reconstruct SpLoRA sparse matrix
            # PyTorch nn.Linear expects W of shape [out_features, in_features]
            # S_sparse_idx maps to the flattened W. Our W_T_approx is transposed [in, out]
            # We reconstruct W, then transpose it.
            S_sparse_dense = torch.zeros(self.out_features * self.in_features, device=X.device, dtype=X.dtype)
            S_sparse_dense.scatter_(0, self.S_sparse_idx.to(torch.int64), self.S_sparse_val.to(X.dtype))
            S_sparse_dense = S_sparse_dense.view(self.out_features, self.in_features).t().contiguous()
            
            W_T_approx = MPO_scaled + LORA + S_sparse_dense
            return torch.matmul(X, W_T_approx)
            
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


def replace_with_qtensor(module, chi, lora_rank, lora_alpha, target_suffixes=None):
    if target_suffixes is None:
        target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(name.endswith(suffix) for suffix in target_suffixes):
            in_features = child.in_features
            out_features = child.out_features
            qtensor_layer = QTensorLinear(in_features, out_features, chi=chi, lora_rank=lora_rank, lora_alpha=lora_alpha)
            setattr(module, name, qtensor_layer)
        else:
            replace_with_qtensor(child, chi, lora_rank, lora_alpha, target_suffixes)
    return module


class QTensorLlamaForCausalLM(LlamaForCausalLM):
    config_class = QTensorLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self = replace_with_qtensor(self, chi=config.qtensor_chi, lora_rank=config.qtensor_lora_rank, lora_alpha=config.qtensor_lora_alpha)
        # Ensure tie weights runs if necessary
        self.post_init()

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64}, num_warps=4),
    ],
    key=['M']
)
@triton.jit
def kronecker_forward_kernel(
    X_ptr, A_ptr, B_ptr, Y_ptr,
    M, 
    m1: tl.constexpr, m2: tl.constexpr, 
    n1: tl.constexpr, n2: tl.constexpr,
    stride_xm, stride_ym,
    BLOCK_M1: tl.constexpr, BLOCK_M2: tl.constexpr, 
    BLOCK_N1: tl.constexpr, BLOCK_N2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    k_dim: tl.constexpr
):
    pid = tl.program_id(0)
    start_m = pid * BLOCK_SIZE_M
    
    offs_n1 = tl.arange(0, BLOCK_N1)
    offs_m1 = tl.arange(0, BLOCK_M1)
    mask_a = (offs_n1[:, None] < n1) & (offs_m1[None, :] < m1)
    
    offs_m2 = tl.arange(0, BLOCK_M2)
    offs_n2 = tl.arange(0, BLOCK_N2)
    mask_b = (offs_m2[:, None] < m2) & (offs_n2[None, :] < n2)
    
    # Process up to BLOCK_SIZE_M rows
    for m in range(start_m, start_m + BLOCK_SIZE_M):
        mask = m < M
        # broadcast mask to the shape of X_mat
        mask_x = mask & (offs_m1[:, None] < m1) & (offs_m2[None, :] < m2)
        
        # Load X[m] of shape (BLOCK_M1, BLOCK_M2)
        X_ptrs = X_ptr + m * stride_xm + offs_m1[:, None] * m2 + offs_m2[None, :] * 1
        X_mat = tl.load(X_ptrs, mask=mask_x, other=0.0) 
        
        Y_acc = tl.zeros((BLOCK_N1, BLOCK_N2), dtype=tl.float32)
        
        for i in range(k_dim):
            # A_T is shape (k_dim, n1, m1)
            # We want A_T[i]
            A_T_ptrs_i = A_ptr + i * (m1 * n1) + offs_n1[:, None] * 1 + offs_m1[None, :] * n1
            A_T_i = tl.load(A_T_ptrs_i, mask=mask_a, other=0.0)
            
            # B is shape (k_dim, m2, n2)
            B_ptrs_i = B_ptr + i * (m2 * n2) + offs_m2[:, None] * n2 + offs_n2[None, :] * 1
            B_mat_i = tl.load(B_ptrs_i, mask=mask_b, other=0.0)
            
            # Z = X @ B_i -> (BLOCK_M1, BLOCK_N2)
            Z = tl.dot(X_mat, B_mat_i, allow_tf32=True)
            Z_bf16 = Z.to(tl.bfloat16)
            
            # Y_acc += A_i^T @ Z -> (BLOCK_N1, BLOCK_N2)
            Y_acc += tl.dot(A_T_i, Z_bf16, allow_tf32=True)
            
        Y = Y_acc.to(tl.bfloat16)
        
        # Store Y[m] of shape (n1, n2)
        mask_y = mask & (offs_n1[:, None] < n1) & (offs_n2[None, :] < n2)
        Y_ptrs = Y_ptr + m * stride_ym + offs_n1[:, None] * n2 + offs_n2[None, :] * 1
        tl.store(Y_ptrs, Y, mask=mask_y)

def fused_kronecker_forward(X, A, B, m1=64, m2=64, n1=64, n2=64, k=1):
    """
    Computes Y = X(sum_i A_i ⊗ B_i) via SRAM fusion.
    X: (M, m1*m2)
    A: (k, m1, n1)
    B: (k, m2, n2)
    Returns Y: (M, n1*n2) in bfloat16
    """
    assert X.is_contiguous()
    M = X.shape[0]
    
    Y = torch.empty((M, n1 * n2), device=X.device, dtype=torch.bfloat16)
    
    A_bf16 = A.to(torch.bfloat16)
    B_bf16 = B.to(torch.bfloat16)
    X_bf16 = X.to(torch.bfloat16)
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
        
    kronecker_forward_kernel[grid](
        X_bf16, A_bf16, B_bf16, Y,
        M, m1, m2, n1, n2,
        X_bf16.stride(0), Y.stride(0),
        BLOCK_M1=max(16, triton.next_power_of_2(m1)),
        BLOCK_M2=max(16, triton.next_power_of_2(m2)),
        BLOCK_N1=max(16, triton.next_power_of_2(n1)),
        BLOCK_N2=max(16, triton.next_power_of_2(n2)),
        k_dim=k
    )
    
    return Y

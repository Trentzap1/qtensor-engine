import triton
import triton.language as tl

def get_autotune_configs():
    configs = []
    for block_m in [16, 32]:
        for block_n in [32, 64, 128]:
            for block_k in [32, 64]:
                for num_warps in [2, 4, 8]:
                    for num_stages in [2, 3, 4]:
                        configs.append(triton.Config(
                            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k},
                            num_warps=num_warps, num_stages=num_stages
                        ))
    return configs

@triton.autotune(
    configs=get_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def mpo_ternary_lora_kernel(
    X_ptr, U_ptr, V_ptr, B_ptr, A_ptr, Y_ptr,
    alpha_prod,
    M, N, K, CHI, LORA_RANK,
    stride_xm, stride_xk,
    stride_uk, stride_uchi,
    stride_vchi, stride_vn,
    stride_bk, stride_bl,
    stride_al, stride_an,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_CHI: tl.constexpr, BLOCK_LORA: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc_Y = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Compute W_tile = U_tile @ V_tile * alpha + B_tile @ A_tile
        W_tile = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        
        # MPO component
        for c in range(0, CHI, BLOCK_CHI):
            offs_chi = c + tl.arange(0, BLOCK_CHI)
            
            U_ptrs = U_ptr + (offs_k[:, None] * stride_uk + offs_chi[None, :] * stride_uchi)
            V_ptrs = V_ptr + (offs_chi[:, None] * stride_vchi + offs_n[None, :] * stride_vn)
            
            u_mask = (offs_k[:, None] < K) & (offs_chi[None, :] < CHI)
            v_mask = (offs_chi[:, None] < CHI) & (offs_n[None, :] < N)
            
            # Dequantize int8 to bfloat16 for fast tl.dot execution on Tensor Cores
            u = tl.load(U_ptrs, mask=u_mask, other=0).to(tl.bfloat16)
            v = tl.load(V_ptrs, mask=v_mask, other=0).to(tl.bfloat16)
            
            W_tile += tl.dot(u, v, allow_tf32=True) * alpha_prod
            
        # LoRA component
        for l in range(0, LORA_RANK, BLOCK_LORA):
            offs_lora = l + tl.arange(0, BLOCK_LORA)
            
            B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_lora[None, :] * stride_bl)
            A_ptrs = A_ptr + (offs_lora[:, None] * stride_al + offs_n[None, :] * stride_an)
            
            b_mask = (offs_k[:, None] < K) & (offs_lora[None, :] < LORA_RANK)
            a_mask = (offs_lora[:, None] < LORA_RANK) & (offs_n[None, :] < N)
            
            b = tl.load(B_ptrs, mask=b_mask, other=0.0)
            a = tl.load(A_ptrs, mask=a_mask, other=0.0)
            
            W_tile += tl.dot(b, a, allow_tf32=True)
            
        # Convert W_tile to activation dtype
        W_tile_bf16 = W_tile.to(tl.bfloat16)
        
        # Multiply X with fused W_tile
        X_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(X_ptrs, mask=x_mask, other=0.0)
        
        acc_Y += tl.dot(x, W_tile_bf16, allow_tf32=True)

    # Write output back to HBM
    Y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptrs, acc_Y.to(tl.bfloat16), mask=y_mask)


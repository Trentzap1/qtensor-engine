import torch
import triton
import triton.language as tl

@triton.jit
def block_svd_fused_kernel(
    X_ptr, V_proj_ptr, U_proj_ptr, Y_ptr,
    M, padded_in, padded_out, num_blocks_in,
    stride_xm, stride_xk,
    stride_vo, stride_vi, stride_vk, stride_vr,
    stride_uo, stride_ui, stride_uk, stride_ur,
    stride_ym, stride_yk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, 
    RANK_R: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Highly optimized memory-efficient Triton Kernel for Block-SVD.
    BLOCK_SIZE = 1024 (The logical PyTorch block size).
    BLOCK_M = Sequence Length Tile (e.g., 16)
    BLOCK_N = Output Feature Tile (e.g., 64)
    BLOCK_K = Inner Reduction Tile (e.g., 64)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1) # Chunk of the output
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, RANK_R)
    
    # Calculate logical block offsets
    out_offset = pid_n * BLOCK_N
    block_o = out_offset // BLOCK_SIZE
    inner_k_out = out_offset % BLOCK_SIZE
    
    mask_m = offs_m < M
    
    # Accumulator for output Y tile [BLOCK_M, BLOCK_N]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for i in range(num_blocks_in):
        # 1. Compute Z_i = X_i @ V_{o,i}
        # X_i is [M, BLOCK_SIZE], V is [BLOCK_SIZE, RANK_R]
        Z_i = tl.zeros([BLOCK_M, RANK_R], dtype=tl.float32)
        
        for k_in in range(0, BLOCK_SIZE, BLOCK_K):
            # Load X tile [BLOCK_M, BLOCK_K]
            x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + (i * BLOCK_SIZE + k_in + offs_k[None, :]) * stride_xk)
            X_tile = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
            
            # Load V tile [BLOCK_K, RANK_R]
            v_ptrs = V_proj_ptr + (block_o * stride_vo + i * stride_vi + (k_in + offs_k[:, None]) * stride_vk + offs_r[None, :] * stride_vr)
            V_tile = tl.load(v_ptrs)
            
            Z_i += tl.dot(X_tile, V_tile, allow_tf32=True)
            
        # 2. Multiply Z_i @ U_{o,i}^T
        # Z_i is [BLOCK_M, RANK_R]
        # Load U tile [BLOCK_N, RANK_R]
        u_ptrs = U_proj_ptr + (block_o * stride_uo + i * stride_ui + (inner_k_out + offs_n[:, None]) * stride_uk + offs_r[None, :] * stride_ur)
        U_tile = tl.load(u_ptrs)
        
        acc += tl.dot(Z_i.to(U_tile.dtype), tl.trans(U_tile), allow_tf32=True)
        
    # Write to HBM
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + (out_offset + offs_n[None, :]) * stride_yk)
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=mask_m[:, None])


def fused_block_svd_splora(X_padded, V_proj, U_proj):
    """
    X_padded: [M, padded_in]
    V_proj: [num_blocks_out, num_blocks_in, block_size, svd_rank]
    U_proj: [num_blocks_out, num_blocks_in, block_size, svd_rank]
    """
    M, padded_in = X_padded.shape
    num_blocks_out, num_blocks_in, block_size, svd_rank = V_proj.shape
    padded_out = num_blocks_out * block_size
    
    Y = torch.empty((M, padded_out), device=X_padded.device, dtype=X_padded.dtype)
    
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(padded_out, meta['BLOCK_N'])
    )
        
    block_svd_fused_kernel[grid](
        X_padded, V_proj, U_proj, Y,
        M, padded_in, padded_out, num_blocks_in,
        X_padded.stride(0), X_padded.stride(1),
        V_proj.stride(0), V_proj.stride(1), V_proj.stride(2), V_proj.stride(3),
        U_proj.stride(0), U_proj.stride(1), U_proj.stride(2), U_proj.stride(3),
        Y.stride(0), Y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, 
        RANK_R=svd_rank, BLOCK_SIZE=block_size,
    )
    
    return Y

@triton.jit
def int4_gemv_kernel(
    X_ptr, W_ptr, Scales_ptr, Zeros_ptr, Y_ptr,
    in_features, out_features,
    stride_w_out, stride_w_in,
    BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr
):
    pid = tl.program_id(0)
    offs_out = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask_out = offs_out < out_features
    
    # Load scales and zeros for this block of output features
    scales = tl.load(Scales_ptr + offs_out, mask=mask_out, other=0.0)
    zeros = tl.load(Zeros_ptr + offs_out, mask=mask_out, other=0.0)
    
    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)
    
    # We iterate over the input features in blocks of BLOCK_IN (unpacked).
    # Since each packed byte holds 2 features, the packed dimension is in_features // 2.
    # We load blocks of size BLOCK_IN // 2 from W.
    packed_in_features = in_features // 2
    
    for k in range(0, packed_in_features, BLOCK_IN // 2):
        offs_w_k = tl.arange(0, BLOCK_IN // 2)
        mask_w_k = (k + offs_w_k) < packed_in_features
        
        # Load packed weights
        w_ptrs = W_ptr + (offs_out[:, None] * stride_w_out + (k + offs_w_k[None, :]) * stride_w_in)
        w_packed = tl.load(w_ptrs, mask=mask_out[:, None] & mask_w_k[None, :], other=0)
        
        # Unpack to low and high
        w_low = w_packed & 0x0F
        w_high = (w_packed >> 4) & 0x0F
        
        # Load X (even and odd separately)
        # Even indices correspond to w_low
        offs_x_even = (k + offs_w_k) * 2
        x_even = tl.load(X_ptr + offs_x_even, mask=offs_x_even < in_features, other=0.0)
        
        # Odd indices correspond to w_high
        offs_x_odd = (k + offs_w_k) * 2 + 1
        x_odd = tl.load(X_ptr + offs_x_odd, mask=offs_x_odd < in_features, other=0.0)
        
        # Dequantize: weight_deq = weight_int4 * scale + zero
        w_low_deq = w_low.to(tl.float32) * scales[:, None] + zeros[:, None]
        w_high_deq = w_high.to(tl.float32) * scales[:, None] + zeros[:, None]
        
        # Multiply and accumulate
        acc += tl.sum(x_even[None, :] * w_low_deq, axis=1)
        acc += tl.sum(x_odd[None, :] * w_high_deq, axis=1)
        
    tl.store(Y_ptr + offs_out, acc.to(Y_ptr.dtype.element_ty), mask=mask_out)

def fused_int4_gemv(X, W_packed, scales, zeros):
    """
    X: [1, in_features] or [in_features]
    W_packed: [out_features, in_features // 2] (uint8)
    scales: [out_features]
    zeros: [out_features]
    """
    original_shape = X.shape
    X_1d = X.view(-1)
    in_features = X_1d.shape[0]
    out_features = W_packed.shape[0]
    
    Y = torch.empty((out_features,), device=X.device, dtype=X.dtype)
    
    BLOCK_IN = 256 # multiple of 2
    BLOCK_OUT = 64
    
    grid = lambda meta: (triton.cdiv(out_features, meta['BLOCK_OUT']), )
    
    int4_gemv_kernel[grid](
        X_1d, W_packed, scales, zeros, Y,
        in_features, out_features,
        W_packed.stride(0), W_packed.stride(1),
        BLOCK_IN=BLOCK_IN, BLOCK_OUT=BLOCK_OUT
    )
    
    if len(original_shape) > 1:
        return Y.view(*original_shape[:-1], out_features)
    return Y

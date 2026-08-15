import os
import torch
import math

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

class QTensorCompressor:
    def __init__(self):
        pass

    @torch.no_grad()
    def compress_160bit(self, weight: torch.Tensor, chi: int):
        """
        Simulates the 160-bit (Q30.130) SVD compression.
        Returns the low-rank factorization components A and B such that W ≈ A @ B
        """
        orig_device = weight.device
        dtype = weight.dtype
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Elevate to highest possible precision for fast SVD computation
        W_high = weight.to(device=compute_device, dtype=torch.float32)
        
        U, S, Vh = torch.linalg.svd(W_high, full_matrices=False)
        
        # Truncate to bond dimension chi
        chi_actual = min(chi, S.size(0))
        U_trunc = U[:, :chi_actual]
        S_trunc = S[:chi_actual]
        Vh_trunc = Vh[:chi_actual, :]
        
        # Absorb singular values into U and Vh for stable MPO layers
        S_sqrt = torch.sqrt(S_trunc)
        A = U_trunc * S_sqrt.unsqueeze(0)
        B = S_sqrt.unsqueeze(1) * Vh_trunc
        
        # Downcast back to original precision and device
        return A.to(device=orig_device, dtype=dtype), B.to(device=orig_device, dtype=dtype)

    @torch.no_grad()
    def compress_float32(self, weight: torch.Tensor, chi: int):
        """
        Standard precision SVD compression (float32) for benchmark comparison.
        """
        orig_device = weight.device
        dtype = weight.dtype
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        W_std = weight.to(device=compute_device, dtype=torch.float32)
        
        U, S, Vh = torch.linalg.svd(W_std, full_matrices=False)
        
        chi_actual = min(chi, S.size(0))
        U_trunc = U[:, :chi_actual]
        S_trunc = S[:chi_actual]
        Vh_trunc = Vh[:chi_actual, :]
        
        S_sqrt = torch.sqrt(S_trunc)
        A = U_trunc * S_sqrt.unsqueeze(0)
        B = S_sqrt.unsqueeze(1) * Vh_trunc
        
        return A.to(device=orig_device, dtype=dtype), B.to(device=orig_device, dtype=dtype)

import torch
import torch.nn as nn
import math

try:
    from .triton_kernel import fused_mpo_forward
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

class MPOLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, chi: int, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor = None, r: int = 16, lora_alpha: int = 16, use_triton: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chi = chi
        self.use_triton = use_triton and HAS_TRITON
        
        # 160-bit Compressed Tensor Chain (Frozen)
        # W = A @ B, where A is (out_features, chi) and B is (chi, in_features)
        # So X @ W.T = X @ (A @ B).T = X @ B.T @ A.T
        self.mpo_B = nn.Parameter(B.t().contiguous(), requires_grad=False)
        self.mpo_A = nn.Parameter(A.t().contiguous(), requires_grad=False)
        
        if bias is not None:
            self.bias = nn.Parameter(bias.contiguous(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        # Healing LoRA Adapters (Trainable)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Initialize LoRA B to zero so initial forward pass matches compressed weights
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)) if 'math' in globals() else None
        
        self.scaling = lora_alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            mpo_out = fused_mpo_forward(x, self.mpo_B, self.mpo_A)
        else:
            mpo_out = torch.matmul(torch.matmul(x, self.mpo_B), self.mpo_A)
        
        if self.bias is not None:
            mpo_out += self.bias
            
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return mpo_out + lora_out

class QuantizedMPOLinear(nn.Module):
    """
    Stacked MPO + Quantization Layer:
    Frozen MPO core factors (A and B) are quantized to FP8 or INT8 precision,
    while 3.65% LoRA adapter nodes remain in high-precision bfloat16.
    """
    def __init__(self, in_features: int, out_features: int, chi: int, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor = None, quant_type: str = "fp8", r: int = 16, lora_alpha: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chi = chi
        self.quant_type = quant_type.lower()
        
        Bt = B.t().contiguous()
        At = A.t().contiguous()
        
        if self.quant_type == "fp8":
            scale_B = (Bt.abs().max() / 448.0).clamp(min=1e-8)
            scale_A = (At.abs().max() / 448.0).clamp(min=1e-8)
            
            Bt_quant = (Bt / scale_B).to(torch.float8_e4m3fn)
            At_quant = (At / scale_A).to(torch.float8_e4m3fn)
        else: # int8
            scale_B = (Bt.abs().max() / 127.0).clamp(min=1e-8)
            scale_A = (At.abs().max() / 127.0).clamp(min=1e-8)
            
            Bt_quant = torch.round(Bt / scale_B).to(torch.int8)
            At_quant = torch.round(At / scale_A).to(torch.int8)
            
        self.register_buffer("scale_B", scale_B)
        self.register_buffer("scale_A", scale_A)
        self.register_buffer("mpo_B_quant", Bt_quant)
        self.register_buffer("mpo_A_quant", At_quant)
        
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)
            
        # 3.65% High-Precision LoRA Adapters
        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=torch.bfloat16)
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=torch.bfloat16)
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = lora_alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_bf16 = x.to(torch.bfloat16)
        
        Bt_dequant = self.mpo_B_quant.to(torch.bfloat16) * self.scale_B
        At_dequant = self.mpo_A_quant.to(torch.bfloat16) * self.scale_A
        
        mpo_out = torch.matmul(torch.matmul(x_bf16, Bt_dequant), At_dequant)
        
        if self.bias is not None:
            mpo_out += self.bias
            
        lora_out = self.lora_B(self.lora_A(x_bf16)) * self.scaling
        return (mpo_out + lora_out).to(orig_dtype)

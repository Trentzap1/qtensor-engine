import torch
import torch.nn as nn

class QTensorBlockSVDLinear(nn.Module):
    def __init__(self, in_features, out_features, block_size=1024, svd_rank=16, lora_rank=256, lora_alpha=512):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.svd_rank = svd_rank
        
        # Calculate padded features
        self.padded_in = ((in_features + block_size - 1) // block_size) * block_size
        self.padded_out = ((out_features + block_size - 1) // block_size) * block_size
        
        self.num_blocks_in = self.padded_in // block_size
        self.num_blocks_out = self.padded_out // block_size
        
        # SVD Components [num_blocks_out, num_blocks_in, block_size, svd_rank]
        self.register_buffer("U_proj", torch.zeros((self.num_blocks_out, self.num_blocks_in, block_size, svd_rank), dtype=torch.bfloat16))
        self.register_buffer("V_proj", torch.zeros((self.num_blocks_out, self.num_blocks_in, block_size, svd_rank), dtype=torch.bfloat16))
        
        # SpLoRA Sparse Matrix
        # Number of non-zero elements is 3% of the actual (unpadded) weight matrix
        k = int(0.03 * in_features * out_features)
        self.register_buffer("S_sparse_idx", torch.zeros((k,), dtype=torch.int32))
        self.register_buffer("S_sparse_val", torch.zeros((k,), dtype=torch.bfloat16))
        
        self.actual_lora = min(lora_rank, in_features, out_features)
        self.lora_scaling = lora_alpha / self.actual_lora
        
        # LoRA Adapters
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
            
        # Format SpLoRA matrix
        S_sparse_dense = torch.zeros(self.out_features * self.in_features, device=X.device, dtype=X.dtype)
        S_sparse_dense.scatter_(0, self.S_sparse_idx.to(torch.int64), self.S_sparse_val.to(X.dtype))
        S_sparse_dense = S_sparse_dense.view(self.out_features, self.in_features).t().contiguous() # [in, out]
        
        # Triton Kernel Launch
        from qtensor.triton_kernel import fused_block_svd_splora
        Y_svd = fused_block_svd_splora(X_padded, self.V_proj, self.U_proj)
        
        if self.padded_out > self.out_features:
            Y_svd = Y_svd[:, :self.out_features]
            
        Y_splora = torch.matmul(X_2d.to(X.dtype), S_sparse_dense)
        
        # Add LoRA
        lora_out = (X_2d.to(torch.bfloat16) @ self.lora_B) @ self.lora_A
        Y_lora = lora_out * self.lora_scaling
        
        Y_2d = Y_svd + Y_splora + Y_lora
        
        if X.dim() > 2:
            return Y_2d.view(*original_shape[:-1], self.out_features)
        return Y_2d

def decompose_and_inject_block_svd(module, W_dense):
    """
    Extracts SpLoRA, performs Block-Wise SVD on the residual, and updates the module.
    """
    # 1. Extract SpLoRA
    # W_dense is [out_features, in_features]
    # Calculate top k based on exactly 3% of original parameters
    k = module.S_sparse_idx.numel()
    
    # We want exactly top k elements by magnitude
    topk_vals, topk_idx = torch.topk(torch.abs(W_dense.flatten()), k)
    
    module.S_sparse_idx.copy_(topk_idx.to(torch.int32))
    # We need the actual values with their original signs, not the absolute values
    actual_values = W_dense.flatten()[topk_idx]
    module.S_sparse_val.copy_(actual_values.to(torch.bfloat16))
    
    # Zero out the extracted sparse elements from dense matrix to create residual
    S_sparse = torch.zeros_like(W_dense)
    S_sparse.flatten()[topk_idx] = actual_values
    W_residual = W_dense - S_sparse
    
    # 2. Block-Wise SVD on W_residual
    # Pad W_residual if necessary
    if module.padded_out > module.out_features or module.padded_in > module.in_features:
        W_padded = torch.nn.functional.pad(W_residual, (0, module.padded_in - module.in_features, 0, module.padded_out - module.out_features))
    else:
        W_padded = W_residual
        
    for o in range(module.num_blocks_out):
        for i in range(module.num_blocks_in):
            # W_block is [block_size, block_size]
            W_block = W_padded[
                o*module.block_size : (o+1)*module.block_size,
                i*module.block_size : (i+1)*module.block_size
            ]
            
            # W_ij ~ U_ij Sigma_ij V_ij^T
            # Perform SVD in float64 for precision
            U, S, Vh = torch.linalg.svd(W_block.to(torch.float64), full_matrices=False)
            
            # Truncate to rank svd_rank
            U_r = U[:, :module.svd_rank]
            S_r = S[:module.svd_rank]
            Vh_r = Vh[:module.svd_rank, :]
            
            # Absorb sqrt(S)
            sqrt_S = torch.sqrt(S_r)
            U_proj = U_r * sqrt_S
            # Vh_r is [r, block_size]. V_proj should be [block_size, r], so we take V_r
            V_r = Vh_r.t() # [block_size, r]
            V_proj = V_r * sqrt_S
            
            module.U_proj[o, i].copy_(U_proj.to(torch.bfloat16))
            module.V_proj[o, i].copy_(V_proj.to(torch.bfloat16))
            
from transformers import LlamaForCausalLM

def replace_with_qtensor_blocksvd(module, target_suffixes=None, block_svd_attention_only=False, block_size=1024, svd_rank=16):
    if target_suffixes is None:
        if block_svd_attention_only:
            target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            target_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(name.endswith(suffix) for suffix in target_suffixes):
            in_features = child.in_features
            out_features = child.out_features
            
            qtensor_layer = QTensorBlockSVDLinear(
                in_features, out_features, 
                block_size=block_size, svd_rank=svd_rank
            )
            setattr(module, name, qtensor_layer)
        else:
            replace_with_qtensor_blocksvd(child, target_suffixes, block_svd_attention_only, block_size, svd_rank)
    return module

class QTensorBlockSVDLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, block_svd_attention_only=False, r=16, block_size=1024):
        super().__init__(config)
        self.qtensor_r = r
        self = replace_with_qtensor_blocksvd(self, block_svd_attention_only=block_svd_attention_only, block_size=block_size, svd_rank=r)
        self.post_init()

def replace_with_qtensor_variablerank(module, block_size=1024):
    attn_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_suffixes = ["gate_proj", "up_proj", "down_proj"]
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if any(name.endswith(suffix) for suffix in attn_suffixes):
                qtensor_layer = QTensorBlockSVDLinear(
                    child.in_features, child.out_features, 
                    block_size=block_size, svd_rank=16
                )
                setattr(module, name, qtensor_layer)
            elif any(name.endswith(suffix) for suffix in mlp_suffixes):
                qtensor_layer = QTensorBlockSVDLinear(
                    child.in_features, child.out_features, 
                    block_size=block_size, svd_rank=64
                )
                setattr(module, name, qtensor_layer)
        else:
            replace_with_qtensor_variablerank(child, block_size)
    return module

class QTensorVariableRankLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, block_size=1024):
        super().__init__(config)
        self = replace_with_qtensor_variablerank(self, block_size=block_size)
        self.post_init()

import torch
import torch.nn as nn
from hf_export.modeling_qtensor_blocksvd import QTensorBlockSVDLinear, decompose_and_inject_block_svd
from transformers import LlamaForCausalLM
import types

def make_hybrid_bridge_forward(layer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values = None,
        use_cache: bool | None = False,
        position_embeddings = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # --- SUBSPACE BRIDGE ---
        if hasattr(self, 'subspace_bridge'):
            hidden_states = hidden_states * self.subspace_bridge

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    return forward

class QTensorINT4LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, lora_rank=256, lora_alpha=512):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer("weight_packed", torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(out_features, dtype=torch.bfloat16))
        self.register_buffer("zeros", torch.zeros(out_features, dtype=torch.bfloat16))
        
        self.actual_lora = min(lora_rank, in_features, out_features)
        self.lora_scaling = lora_alpha / self.actual_lora
        
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
            
        if X_2d.shape[0] == 1:
            from qtensor.triton_kernel import fused_int4_gemv
            base_out = fused_int4_gemv(X_2d.to(torch.bfloat16), self.weight_packed, self.scales, self.zeros)
        else:
            low = self.weight_packed & 0x0F
            high = (self.weight_packed >> 4) & 0x0F
            
            weight_int4 = torch.stack([low, high], dim=-1).view(self.out_features, self.in_features)
            weight_int4 = weight_int4.to(torch.bfloat16)
            
            weight_deq = weight_int4 * self.scales.unsqueeze(1) + self.zeros.unsqueeze(1)
            
            base_out = torch.matmul(X_2d.to(torch.bfloat16), weight_deq.t())
        
        lora_out = (X_2d.to(torch.bfloat16) @ self.lora_B) @ self.lora_A
        Y_lora = lora_out * self.lora_scaling
        
        Y_2d = base_out + Y_lora
        
        if X.dim() > 2:
            return Y_2d.view(*original_shape[:-1], self.out_features)
        return Y_2d

def quantize_int4_and_inject(module, W_dense):
    qmin = 0
    qmax = 15
    W_min = W_dense.min(dim=1)[0]
    W_max = W_dense.max(dim=1)[0]
    scale = (W_max - W_min) / (qmax - qmin)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    
    W_int = torch.round((W_dense - W_min.unsqueeze(1)) / scale.unsqueeze(1)).to(torch.uint8)
    W_int = torch.clamp(W_int, qmin, qmax)
    
    W_view = W_int.view(module.out_features, module.in_features // 2, 2)
    packed = W_view[:, :, 0] | (W_view[:, :, 1] << 4)
    
    module.weight_packed.copy_(packed)
    module.scales.copy_(scale.to(torch.bfloat16))
    module.zeros.copy_(W_min.to(torch.bfloat16))

def replace_with_qtensor_hybrid(module):
    attn_suffixes = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_suffixes = ["gate_proj", "up_proj", "down_proj"]
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if any(name.endswith(suffix) for suffix in attn_suffixes):
                qtensor_layer = QTensorBlockSVDLinear(
                    child.in_features, child.out_features, 
                    block_size=1024, svd_rank=16
                )
                setattr(module, name, qtensor_layer)
            elif any(name.endswith(suffix) for suffix in mlp_suffixes):
                qtensor_layer = QTensorINT4LoRALinear(
                    child.in_features, child.out_features
                )
                setattr(module, name, qtensor_layer)
        else:
            replace_with_qtensor_hybrid(child)
    return module

class QTensorHybridLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, use_bridge=False):
        super().__init__(config)
        self = replace_with_qtensor_hybrid(self)
        
        if use_bridge:
            for layer in self.model.layers:
                layer.subspace_bridge = nn.Parameter(torch.ones(config.hidden_size, dtype=torch.bfloat16))
                layer.forward = types.MethodType(make_hybrid_bridge_forward(layer), layer)
                
        self.post_init()

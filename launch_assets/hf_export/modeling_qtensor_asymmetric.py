import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
import types

from .modeling_qtensor_blocksvd import replace_with_qtensor_blocksvd
from .modeling_qtensor_kronecker import replace_with_qtensor_kronecker

def make_asymmetric_forward(layer):
    """
    Creates a new forward method for LlamaDecoderLayer that includes
    the Rank-1 Subspace Bridge after the attention residual addition.
    """
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
        hidden_states = hidden_states * self.subspace_bridge

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
    return forward

class QTensorAsymmetricLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, block_size=1024, r=16, kronecker_chi=256, lora_rank=256, lora_alpha=512):
        super().__init__(config)
        self.qtensor_r = r
        self.qtensor_kronecker_chi = kronecker_chi
        
        # 1. Replace Attention layers with Block-Wise SVD
        self = replace_with_qtensor_blocksvd(
            self, 
            target_suffixes=["q_proj", "k_proj", "v_proj", "o_proj"], 
            block_svd_attention_only=True, 
            block_size=block_size, 
            r=r
        )
        
        # 2. Replace MLP layers with Global Kronecker Factorization
        self = replace_with_qtensor_kronecker(
            self,
            kronecker_attention_only=False,
            target_suffixes=["gate_proj", "up_proj", "down_proj"]
        )
        
        # 3. Inject Rank-1 Subspace Bridge into every Decoder Layer
        for layer in self.model.layers:
            layer.subspace_bridge = nn.Parameter(torch.ones(config.hidden_size, dtype=torch.bfloat16))
            # Monkey-patch the forward method
            layer.forward = types.MethodType(make_asymmetric_forward(layer), layer)
            
        self.post_init()

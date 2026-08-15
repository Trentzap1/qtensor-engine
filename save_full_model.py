import os
import torch
from transformers import AutoModelForCausalLM
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from hf_export.configuration_qtensor import QTensorLlamaConfig
from hf_export.modeling_qtensor_hybrid import QTensorHybridLlamaForCausalLM, quantize_int4_and_inject
from hf_export.modeling_qtensor_blocksvd import decompose_and_inject_block_svd

def main():
    print("Initializing full compressed model...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    
    print("Instantiating QTensor Hybrid Student...")
    config = QTensorLlamaConfig.from_pretrained(model_id)
    config.qtensor_chi = 256
    config.qtensor_lora_rank = 256
    config.qtensor_lora_alpha = 512
    if not hasattr(config, "mlp_bias"): config.mlp_bias = False
    
    student = QTensorHybridLlamaForCausalLM(config, use_bridge=True)
    
    print("Loading base weights...")
    student.load_state_dict(teacher.state_dict(), strict=False)
    student = student.to(torch.bfloat16).to(device)
    
    for layer in student.model.layers:
        if hasattr(layer.self_attn, 'rotary_emb') and hasattr(layer.self_attn.rotary_emb, 'inv_freq'):
            layer.self_attn.rotary_emb.inv_freq = layer.self_attn.rotary_emb.inv_freq.to(torch.float32)
            
    print("Decomposing Teacher into Hybrid Architecture...")
    for (name_s, module_s), (name_t, module_t) in zip(student.named_modules(), teacher.named_modules()):
        if hasattr(module_s, 'block_size'):
            decompose_and_inject_block_svd(module_s, module_t.weight.detach())
        elif hasattr(module_s, 'weight_packed'):
            quantize_int4_and_inject(module_s, module_t.weight.detach())
            
    print("Loading QAD LoRA Adapters & Bridge...")
    from safetensors.torch import load_file
    lora_path = "checkpoints/qtensor_tinyllama_qad_hybrid_bridge.safetensors"
    lora_state = load_file(lora_path)
    student.load_state_dict(lora_state, strict=False)
    
    print("Saving FULL merged state_dict to hf_export_hub...")
    student.save_pretrained("hf_export_hub", safe_serialization=True)
    
    print("Done! The full model.safetensors is ready to upload.")

if __name__ == "__main__":
    main()

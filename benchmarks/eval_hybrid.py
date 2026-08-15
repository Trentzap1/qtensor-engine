import os
import sys
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    print("\n--- Evaluating Hybrid Bridge Architecture (HellaSwag & WikiText) ---")
    
    # 1. Model Loading
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Teacher for Decomposition...")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    print("Instantiating QTensor Hybrid Student...", flush=True)
    from hf_export.configuration_qtensor import QTensorLlamaConfig
    from hf_export.modeling_qtensor_hybrid import QTensorHybridLlamaForCausalLM, quantize_int4_and_inject
    from hf_export.modeling_qtensor_blocksvd import decompose_and_inject_block_svd
    
    config = QTensorLlamaConfig.from_pretrained(model_id)
    config.qtensor_chi = 256
    config.qtensor_lora_rank = 256
    config.qtensor_lora_alpha = 512
    if not hasattr(config, "mlp_bias"): config.mlp_bias = False
    
    student = QTensorHybridLlamaForCausalLM(config, use_bridge=True)
    
    print("Loading base weights from Teacher...", flush=True)
    student.load_state_dict(teacher.state_dict(), strict=False)
    
    student = student.to(torch.bfloat16).to(device)
    
    print("Decomposing Teacher weights into Hybrid Subspaces...", flush=True)
    for (name_s, module_s), (name_t, module_t) in zip(student.named_modules(), teacher.named_modules()):
        if hasattr(module_s, 'block_size'):
            decompose_and_inject_block_svd(module_s, module_t.weight.detach())
        elif hasattr(module_s, 'weight_packed'):
            quantize_int4_and_inject(module_s, module_t.weight.detach())
            
    from safetensors.torch import load_file
    lora_path = os.path.join(project_root, "checkpoints", "qtensor_tinyllama_qad_hybrid_bridge.safetensors")
    if os.path.exists(lora_path):
        res = student.load_state_dict(load_file(lora_path), strict=False)
        print(f"LoRA loaded successfully: {res}", flush=True)
        
    student.eval()
    del teacher
    torch.cuda.empty_cache()
    
    print("Wrapping in HFLM...")
    lm = HFLM(pretrained=student, tokenizer=model_id, batch_size=16, max_length=2048)
    
    print("Running lm_eval (HellaSwag & WikiText)...")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["hellaswag", "wikitext"],
        num_fewshot=0,
        batch_size=16,
        limit=200 # Using a limit of 200 for rapid iteration. We can run the full eval if needed.
    )
    
    print("\n--- Results (Subset: 200 samples) ---")
    print("HellaSwag Acc: ", results["results"].get("hellaswag", {}).get("acc,none", 0.0))
    
    wt = results["results"].get("wikitext", {})
    if wt:
        print("WikiText word_perplexity: ", wt.get("word_perplexity,none", "N/A"))
        print("WikiText byte_perplexity: ", wt.get("byte_perplexity,none", "N/A"))
        print("WikiText bits_per_byte: ", wt.get("bits_per_byte,none", "N/A"))
    else:
        print("WikiText results:", results["results"])

if __name__ == "__main__":
    main()

import os
import torch
import time
import sys
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch._inductor.config
import torch._dynamo
torch._dynamo.config.disable = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hf_export.configuration_qtensor import QTensorLlamaConfig
from hf_export.modeling_qtensor_blocksvd import QTensorVariableRankLlamaForCausalLM, decompose_and_inject_block_svd
from hf_export.modeling_qtensor_hybrid import QTensorHybridLlamaForCausalLM, quantize_int4_and_inject
from local_rag.rag_pipeline import RAGPipeline

def init_hybrid_model():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Teacher for Decomposition...")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    print("Instantiating QTensor Hybrid Student...", flush=True)
    config = QTensorLlamaConfig.from_pretrained(model_id)
    config.qtensor_chi = 256
    config.qtensor_lora_rank = 256
    config.qtensor_lora_alpha = 512
    if not hasattr(config, "mlp_bias"): config.mlp_bias = False
    
    student = QTensorHybridLlamaForCausalLM(config, use_bridge=True)
    
    print("Loading base weights (Embeddings, LayerNorms) from Teacher...", flush=True)
    student.load_state_dict(teacher.state_dict(), strict=False)
    
    print("Casting to bfloat16...", flush=True)
    student = student.to(torch.bfloat16)
    print(f"Moving to {device}...", flush=True)
    student = student.to(device)
    
    print("Restoring RoPE inv_freq to float32 to prevent positional collapse...", flush=True)
    for layer in student.model.layers:
        if hasattr(layer.self_attn, 'rotary_emb') and hasattr(layer.self_attn.rotary_emb, 'inv_freq'):
            layer.self_attn.rotary_emb.inv_freq = layer.self_attn.rotary_emb.inv_freq.to(torch.float32)
    
    print("Decomposing Teacher weights into Hybrid Subspaces for Student...", flush=True)
    decomp_idx = 0
    for (name_s, module_s), (name_t, module_t) in zip(student.named_modules(), teacher.named_modules()):
        if hasattr(module_s, 'block_size'):
            if decomp_idx % 20 == 0:
                print(f"Decomposing Block-SVD {name_s} (idx: {decomp_idx})...", flush=True)
            decompose_and_inject_block_svd(module_s, module_t.weight.detach())
            decomp_idx += 1
        elif hasattr(module_s, 'weight_packed'):
            if decomp_idx % 20 == 0:
                print(f"Quantizing INT4 {name_s} (idx: {decomp_idx})...", flush=True)
            quantize_int4_and_inject(module_s, module_t.weight.detach())
            decomp_idx += 1
            
    print(f"Decomposition complete. Replaced {decomp_idx} linear layers.", flush=True)
    
    print("Loading QAD LoRA Adapters to restore syntax and grammar...", flush=True)
    from safetensors.torch import load_file
    lora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "qtensor_tinyllama_qad_hybrid_bridge.safetensors"))
    if os.path.exists(lora_path):
        lora_state = load_file(lora_path)
        res = student.load_state_dict(lora_state, strict=False)
        print(f"LoRA loaded successfully: {res}", flush=True)
    else:
        print(f"WARNING: LoRA adapters not found at {lora_path}", flush=True)
        
    student.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Clean up teacher to free VRAM for an accurate footprint reading
    del teacher
    torch.cuda.empty_cache()
    
    return student, tokenizer, device

def main():
    print("=== Aegis Local Document AI ===")
    
    start_time = time.time()
    
    # 1. Init RAG Pipeline
    pipeline = RAGPipeline()
    
    # 2. Init Model
    model, tokenizer, device = init_hybrid_model()
    
    # Record footprint
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**2)
    print(f"VRAM Allocated: {allocated:.2f} MB")
    
    load_time = time.time() - start_time
    print(f"System loaded in {load_time:.2f} seconds.")
    
    # 3. Interactive Loop / Automated Run
    print("\nSystem ready. Running automated test query...")
    query = "What is the aggregate consideration to be paid for the equity interests, and what is the break-up fee?"
    print(f"\nQuestion: {query}")
        
    print("Retrieving context...")
    retrieval_start = time.time()
    context = pipeline.retrieve_context(query)
    print(f"[Context retrieved in {time.time() - retrieval_start:.2f}s]")
    
    prompt = pipeline.format_prompt(query, context)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("Setting up Native CUDA Graph wrapper...")
    class CudaGraphWrapper:
        def __init__(self, model):
            self.model = model
            self.graph = None
            self.static_kwargs = {}
            self.static_args = []
            self.static_out = None
            self.original_forward = model.forward
            self.stream = torch.cuda.Stream()

        def __call__(self, *args, **kwargs):
            input_ids = kwargs.get('input_ids', args[0] if len(args) > 0 else None)
            
            if input_ids is not None and input_ids.shape[1] > 1:
                return self.original_forward(*args, **kwargs)
                
            if self.graph is None:
                # print("Capturing CUDA Graph...", flush=True)
                self.static_args = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
                self.static_kwargs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
                
                # Save cumulative length before warmup
                past_kv = self.static_kwargs.get("past_key_values", self.static_args[3] if len(self.static_args) > 3 else None)
                saved_states = []
                if past_kv is not None and hasattr(past_kv, "layers"):
                    for layer in past_kv.layers:
                        state = {}
                        if hasattr(layer, "cumulative_length"):
                            state["tensor"] = layer.cumulative_length.clone()
                        if hasattr(layer, "cumulative_length_int"):
                            state["int"] = layer.cumulative_length_int
                        saved_states.append(state)

                # Warmup
                self.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream):
                    for _ in range(3):
                        self.original_forward(*self.static_args, **self.static_kwargs)
                torch.cuda.current_stream().wait_stream(self.stream)
                
                # Restore cumulative length
                if past_kv is not None and hasattr(past_kv, "layers"):
                    for layer, state in zip(past_kv.layers, saved_states):
                        if "tensor" in state:
                            layer.cumulative_length.copy_(state["tensor"])
                        if "int" in state:
                            layer.cumulative_length_int = state["int"]
                
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.static_out = self.original_forward(*self.static_args, **self.static_kwargs)
                
            # Replay phase
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and k in self.static_kwargs and self.static_kwargs[k] is not None:
                    self.static_kwargs[k].copy_(v)
            for i, v in enumerate(args):
                if isinstance(v, torch.Tensor):
                    self.static_args[i].copy_(v)
                
            self.graph.replay()
            return self.static_out

    model.forward = CudaGraphWrapper(model)

    print(f"\nPrompt length: {inputs.input_ids.shape[1]}")
    with torch.no_grad():
        model.generate(
            inputs.input_ids,
            max_new_tokens=5,
            cache_implementation="static",
            pad_token_id=tokenizer.eos_token_id
        )
        
    print("\nReasoning...")
    # Clear the CUDA Graph so it captures the new StaticCache instance!
    if hasattr(model.forward, "graph"):
        model.forward.graph = None
    gen_start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            cache_implementation="static",
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4
        )
        
    gen_time = time.time() - gen_start
    
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("\n" + "="*50)
    # Safe print for Windows console
    safe_response = response.encode('cp1252', errors='replace').decode('cp1252')
    print(safe_response.strip())
    print("="*50)
    
    # Also save to file just in case
    with open("rag_output.txt", "w", encoding="utf-8") as f:
        f.write(response)
    
    num_tokens = len(generated_tokens)
    print(f"[Generated {num_tokens} tokens in {gen_time:.2f}s ({num_tokens/gen_time:.2f} tokens/sec)]")

if __name__ == "__main__":
    main()

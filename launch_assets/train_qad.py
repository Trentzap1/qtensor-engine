import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_wise", action="store_true", help="Use block-wise Kronecker decomposition")
    parser.add_argument("--block_svd", action="store_true", help="Use block-wise SVD + SpLoRA architecture")
    parser.add_argument("--asymmetric", action="store_true", help="Use Asymmetric Rank-1 Subspace Bridge architecture")
    parser.add_argument("--unified_block_svd", action="store_true", help="Use Unified Block-Wise SVD + SpLoRA across all layers")
    parser.add_argument("--variable_rank_svd", action="store_true", help="Use Variable-Rank Block-SVD (Attn r=16, MLP r=64)")
    parser.add_argument("--hybrid", action="store_true", help="Use Hybrid Architecture (BlockSVD Attention + INT4 MLP)")
    parser.add_argument("--hybrid_bridge", action="store_true", help="Use Hybrid Architecture + Subspace Bridge")
    args = parser.parse_args()
    
    print("--- QTensor Quantization-Aware Distillation (QAD) ---")
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    # 1. Model Loading & Freezing
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda:0"
    
    print("Loading FP16 Teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    print("Loading QTensor Student...")
    from hf_export.modeling_qtensor_kronecker import QTensorKroneckerLlamaForCausalLM, decompose_and_inject_kronecker, QTensorKroneckerLinear
    from hf_export.modeling_qtensor_blocksvd import QTensorBlockSVDLlamaForCausalLM, QTensorVariableRankLlamaForCausalLM
    from hf_export.modeling_qtensor_asymmetric import QTensorAsymmetricLlamaForCausalLM
    from hf_export.modeling_qtensor_hybrid import QTensorHybridLlamaForCausalLM, quantize_int4_and_inject
    from hf_export.configuration_qtensor import QTensorLlamaConfig
    
    config = QTensorLlamaConfig.from_pretrained(model_id)
    config.qtensor_chi = 256
    config.qtensor_lora_rank = 256
    config.qtensor_lora_alpha = 512
    if not hasattr(config, "mlp_bias"): config.mlp_bias = False
    if not hasattr(config, "attention_bias"): config.attention_bias = False
    
    print("Instantiating QTensor student...")
    if args.hybrid_bridge:
        print("Using Hybrid Architecture + Subspace Bridge (BlockSVD Attention + INT4 MLP)")
        student = QTensorHybridLlamaForCausalLM(config, use_bridge=True).to(torch.bfloat16).to(device)
    elif args.hybrid:
        print("Using Hybrid Architecture (BlockSVD Attention + INT4 MLP)")
        student = QTensorHybridLlamaForCausalLM(config, use_bridge=False).to(torch.bfloat16).to(device)
    elif args.variable_rank_svd:
        print("Using Variable-Rank Block-SVD (Attention r=16, MLP r=64)")
        student = QTensorVariableRankLlamaForCausalLM(config, block_size=1024).to(torch.bfloat16).to(device)
    elif args.unified_block_svd:
        print("Using Unified Block-Wise SVD + SpLoRA (r=16, block_size=1024) across ALL layers")
        student = QTensorBlockSVDLlamaForCausalLM(config, block_svd_attention_only=False, r=16, block_size=1024).to(torch.bfloat16).to(device)
    elif args.asymmetric:
        print("Using Asymmetric Subspace Bridge (BlockSVD Attention + Kronecker MLP)")
        student = QTensorAsymmetricLlamaForCausalLM(config, block_size=1024, r=16, kronecker_chi=256).to(torch.bfloat16).to(device)
    elif args.block_svd:
        print("Using Block-Wise SVD + SpLoRA (r=16, block_size=1024)")
        student = QTensorBlockSVDLlamaForCausalLM(config, block_svd_attention_only=True, r=16, block_size=1024).to(torch.bfloat16).to(device)
    elif args.block_wise:
        print("Using Block-Wise Kronecker Factorization (k=1, block_size=1024)")
        student = QTensorKroneckerLlamaForCausalLM(config, kronecker_attention_only=True, kronecker_k=1, block_size=1024).to(torch.bfloat16).to(device)
    else:
        student = QTensorKroneckerLlamaForCausalLM(config).to(torch.bfloat16).to(device)
        
    print("Loading base weights (Embeddings, LayerNorms, MLPs) from Teacher...", flush=True)
    student.load_state_dict(teacher.state_dict(), strict=False)
        
    print("Restoring RoPE inv_freq to float32 to prevent positional collapse...", flush=True)
    for layer in student.model.layers:
        if hasattr(layer.self_attn, 'rotary_emb') and hasattr(layer.self_attn.rotary_emb, 'inv_freq'):
            layer.self_attn.rotary_emb.inv_freq = layer.self_attn.rotary_emb.inv_freq.to(torch.float32)
    
    print("Decomposing Teacher weights into compressed Factors for Student...")
    from hf_export.modeling_qtensor_kronecker import decompose_and_inject_block_kronecker
    from hf_export.modeling_qtensor_blocksvd import decompose_and_inject_block_svd
    decomp_idx = 0
    for (name_s, module_s), (name_t, module_t) in zip(student.named_modules(), teacher.named_modules()):
        if args.asymmetric:
            if hasattr(module_s, 'block_size') and not isinstance(module_s, QTensorKroneckerLinear):
                print(f"Decomposing Asymmetric Block-Wise SVD {name_s} (idx: {decomp_idx}) ...", flush=True)
                decompose_and_inject_block_svd(module_s, module_t.weight.detach())
                decomp_idx += 1
            elif isinstance(module_s, QTensorKroneckerLinear):
                print(f"Decomposing Asymmetric Global Kronecker {name_s} (idx: {decomp_idx}) ...", flush=True)
                decompose_and_inject_kronecker(module_s, module_t.weight.detach())
                decomp_idx += 1
        else:
            if args.hybrid or args.hybrid_bridge:
                if hasattr(module_s, 'block_size'):
                    print(f"Decomposing Hybrid Block-SVD {name_s} (idx: {decomp_idx}) ...", flush=True)
                    decompose_and_inject_block_svd(module_s, module_t.weight.detach())
                    decomp_idx += 1
                elif hasattr(module_s, 'weight_packed'):
                    print(f"Quantizing Hybrid INT4 {name_s} (idx: {decomp_idx}) ...", flush=True)
                    quantize_int4_and_inject(module_s, module_t.weight.detach())
                    decomp_idx += 1
            elif args.variable_rank_svd and hasattr(module_s, 'block_size'):
                print(f"Decomposing Variable-Rank Block-SVD {name_s} (idx: {decomp_idx}) ...", flush=True)
                decompose_and_inject_block_svd(module_s, module_t.weight.detach())
                decomp_idx += 1
            elif args.unified_block_svd and hasattr(module_s, 'block_size') and not isinstance(module_s, QTensorKroneckerLinear):
                print(f"Decomposing Unified Block-Wise SVD {name_s} (idx: {decomp_idx}) ...", flush=True)
                decompose_and_inject_block_svd(module_s, module_t.weight.detach())
                decomp_idx += 1
            elif args.block_svd and hasattr(module_s, 'block_size') and not isinstance(module_s, QTensorKroneckerLinear):
                print(f"Decomposing Block-Wise SVD {name_s} (idx: {decomp_idx}) ...", flush=True)
                decompose_and_inject_block_svd(module_s, module_t.weight.detach())
                decomp_idx += 1
            elif args.block_wise and hasattr(module_s, 'block_size'):
                print(f"Decomposing Block-Wise Kronecker {name_s} (idx: {decomp_idx}) ...", flush=True)
                decompose_and_inject_block_kronecker(module_s, module_t.weight.detach())
                decomp_idx += 1
            elif not args.block_wise and not args.block_svd and isinstance(module_s, QTensorKroneckerLinear):
                print(f"Decomposing Global {name_s} (idx: {decomp_idx}) ...", flush=True)
                decompose_and_inject_kronecker(module_s, module_t.weight.detach())
                decomp_idx += 1
    print("Decomposition complete.", flush=True)
        
    student.train()
    
    # Freeze everything except LoRA adapters and subspace bridges
    for name, param in student.named_parameters():
        if "lora" in name.lower() or "subspace_bridge" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 2. Dataset & Processing
    print("Loading Alpaca dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    def format_and_tokenize(examples):
        instructions = examples.get('instruction', [])
        inputs = examples.get('input', [])
        outputs = examples.get('output', [])
        
        batch_input_ids = []
        batch_attention_mask = []
        for inst, inp, out in zip(instructions, inputs, outputs):
            prompt = f"Instruction: {inst}\n"
            if inp:
                prompt += f"Input: {inp}\n"
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": out}
            ]
            
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized = tokenizer(
                formatted, 
                truncation=True, 
                max_length=512, 
                padding="max_length"
            )
            batch_input_ids.append(tokenized["input_ids"])
            batch_attention_mask.append(tokenized["attention_mask"])
            
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask
        }
        
    print("Tokenizing dataset...", flush=True)
    dataset = dataset.map(format_and_tokenize, batched=True, num_proc=8, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. Training Loop & Loss Computation
    learning_rate = 2e-4
    T = 2.0
    grad_accum_steps = 8
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader)//grad_accum_steps, eta_min=1e-5)
    
    print("Starting Full QAD Run...")
    
    step = 0
    total_loss = 0.0
    
    optimizer.zero_grad()
    out_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    from safetensors.torch import save_file
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Teacher Forward
        with torch.no_grad():
            outputs_teacher = teacher(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            z_teacher = outputs_teacher.logits
            
        # Student Forward
        outputs_student = student(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        z_student = outputs_student.logits
        
        vocab_size = z_student.size(-1)
        z_student_flat = z_student.view(-1, vocab_size)
        z_teacher_flat = z_teacher.view(-1, vocab_size)
        
        # Distillation Loss
        loss_kl = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(z_student_flat / T, dim=-1),
            F.softmax(z_teacher_flat / T, dim=-1)
        ) * (T ** 2)
        
        # Hidden-State MSE Loss
        anchor_layers = [4, 8, 12, 16, 20, 22]
        mse_loss = 0.0
        for l in anchor_layers:
            student_h = outputs_student.hidden_states[l]
            teacher_h = outputs_teacher.hidden_states[l].detach()
            mse_loss += torch.nn.functional.mse_loss(student_h, teacher_h)
        mse_loss = mse_loss / len(anchor_layers)
        
        loss = loss_kl + mse_loss
        
        loss = loss / grad_accum_steps
        loss.backward()
        
        total_loss += loss.item() * grad_accum_steps
        
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            actual_step = (step + 1) // grad_accum_steps
            if actual_step % 10 == 0:
                print(f"Step {actual_step} | Total Loss (KL+MSE): {total_loss / 10:.4f}", flush=True)
                total_loss = 0.0
                
            if actual_step % 250 == 0:
                out_path = os.path.join(out_dir, f"qtensor_tinyllama_qad_step_{actual_step}.safetensors")
                save_dict = {k: v for k, v in student.state_dict().items() if "lora" in k.lower() or "subspace_bridge" in k.lower()}
                save_file(save_dict, out_path)
                print(f"Saved checkpoint to {out_path}", flush=True)
                
            if actual_step >= 500:
                print("Reached 500 steps. Stopping early for evaluation.", flush=True)
                break
                
        step += 1
        
        if (step // grad_accum_steps) >= 500:
            break

    # 4. Final Checkpointing
    if args.hybrid_bridge:
        out_name = "qtensor_tinyllama_qad_hybrid_bridge.safetensors"
    elif args.hybrid:
        out_name = "qtensor_tinyllama_qad_hybrid.safetensors"
    elif args.variable_rank_svd:
        out_name = "qtensor_tinyllama_qad_variablerank.safetensors"
    elif args.unified_block_svd:
        out_name = "qtensor_tinyllama_qad_unified_blocksvd.safetensors"
    elif args.asymmetric:
        out_name = "qtensor_tinyllama_qad_asymmetric.safetensors"
    elif args.block_svd:
        out_name = "qtensor_tinyllama_qad_blocksvd.safetensors"
    elif args.block_wise:
        out_name = "qtensor_tinyllama_qad_blockwise.safetensors"
    else:
        out_name = "qtensor_tinyllama_qad_final.safetensors"
    out_path = os.path.join(out_dir, out_name)
    save_dict = {k: v for k, v in student.state_dict().items() if "lora" in k.lower() or "subspace_bridge" in k.lower()}
    save_file(save_dict, out_path)
    print(f"Full Distillation complete. Final LoRA adapters saved to {out_path}")

if __name__ == "__main__":
    main()

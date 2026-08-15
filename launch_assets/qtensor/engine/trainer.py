import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from qtensor.engine.patcher import replace_with_qtensor, load_qtensor_weights
import os
from tqdm import tqdm
from torch.optim import AdamW

os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"

def get_dataloader(tokenizer, seq_len=256, batch_size=4):
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    def generate_batches():
        batch = []
        buffer = []
        for example in dataset:
            tokens = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            
            while len(buffer) >= seq_len:
                batch.append(buffer[:seq_len])
                buffer = buffer[seq_len:]
                
                if len(batch) == batch_size:
                    yield torch.tensor(batch, dtype=torch.long)
                    batch = []

    return generate_batches()

def eval_ppl(model, tokenizer, device, max_tokens=10000):
    model.eval()
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    
    seq_len = encodings.input_ids.size(1)
    stride = 512
    max_length = 2048
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, min(seq_len, max_tokens), stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
            
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    ppl = torch.exp(torch.stack(nlls).mean())
    model.train()
    return ppl.item()

def generate_text_test(model, tokenizer, device, prompt="The capital of Australia is"):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[Generation Test]:\n{text}\n")
    model.train()

def train_healing_loop():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    safetensors_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtensor_tinyllama_1.1b.safetensors")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtensor_tinyllama_1.1b_healed.safetensors")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("1. Loading Teacher Model (Dense FP16)...")
    teacher = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    print("2. Loading Student Model (QTensor)...")
    config = AutoConfig.from_pretrained(model_id)
    student = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    student = replace_with_qtensor(student, chi=256, lora_rank=128)
    load_qtensor_weights(student, safetensors_path)
    student.to(device)
    student.train()
    
    print("3. Freezing MPO Cores & Unfreezing LoRA Adapters...")
    trainable_params = []
    for name, param in student.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total Trainable LoRA Parameters: {total_trainable:,}")
    
    max_steps = 10000
    optimizer = AdamW(trainable_params, lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=max_steps)
    dataloader = get_dataloader(tokenizer, seq_len=256, batch_size=4)
    
    print("\n--- Starting QAT Healing (10,000 Steps) ---")
    step = 0
    pbar = tqdm(total=max_steps)
    
    for batch in dataloader:
        if step >= max_steps:
            break
            
        batch = batch.to(device)
        
        with torch.no_grad():
            teacher_logits = teacher(batch).logits
            
        student_logits = student(batch).logits
        
        s_logits = student_logits.view(-1, student_logits.size(-1))
        t_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        
        loss = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction='batchmean'
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pbar.update(1)
        pbar.set_description(f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        step += 1
        
        if step % 1000 == 0:
            print(f"\nStep {step} - Evaluating Validation Perplexity...")
            ppl = eval_ppl(student, tokenizer, device, max_tokens=10000)
            print(f"--> Step {step} WikiText-2 PPL: {ppl:.2f}")
            generate_text_test(student, tokenizer, device)
            
            # Save checkpoint
            from safetensors.torch import save_file
            ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"checkpoints/qtensor_step_{step}.safetensors")
            final_dict = {}
            for k, v in student.state_dict().items():
                final_dict[k] = v.contiguous()
            save_file(final_dict, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
        
    pbar.close()
    
    print("\nTraining Complete! Saving final healed adapters...")
    from safetensors.torch import save_file
    final_dict = {}
    for k, v in student.state_dict().items():
        final_dict[k] = v.contiguous()
    save_file(final_dict, output_path)
    print(f"Final healed weights saved to {output_path}!")
    
if __name__ == '__main__':
    train_healing_loop()

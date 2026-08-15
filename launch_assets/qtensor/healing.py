import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def heal_model(model: nn.Module, tokenizer: AutoTokenizer, dataset_name: str = "Salesforce/wikitext", config_name: str = "wikitext-2-raw-v1", steps: int = 50, batch_size: int = 2, lr: float = 5e-4):
    """
    Executes post-compression fine-tuning (Healing).
    Freezes MPO tensors and trains 3.65% LoRA adapter parameters on meaningful text blocks.
    """
    
    print("Configuring parameters for healing...")
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "lora_" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            
    print(f"Total Params: {total_params:,} | Trainable Params: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    print(f"Loading dataset: {dataset_name} ({config_name})")
    raw_datasets = load_dataset(dataset_name, config_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Filter non-empty meaningful text
    train_texts = [t for t in raw_datasets["train"]["text"] if len(t.strip()) > 30]
    
    encoded = tokenizer(train_texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}

        def __len__(self):
            return len(self.input_ids)

    dataset = TextDataset(encoded)
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    
    model.train()
    print(f"Starting healing loop for {steps} steps...")
    
    progress_bar = tqdm(range(steps))
    step = 0
    
    for batch in train_dataloader:
        if step >= steps:
            break
            
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # IGNORE PADDING TOKENS IN LOSS
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        step += 1
        
    print("Healing complete!")
    model.eval()
    return model

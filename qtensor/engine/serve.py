import gradio as gr
import torch
import time
import sys
import os
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from safetensors.torch import load_file

# Add hf_export to path to load our custom architecture
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from hf_export.modeling_qtensor import QTensorLlamaForCausalLM
from hf_export.configuration_qtensor import QTensorLlamaConfig

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("Initializing QTensor Triton Architecture...")
device = "cuda" if torch.cuda.is_available() else "cpu"

config = QTensorLlamaConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
config.qtensor_chi = 256
config.qtensor_lora_rank = 128
if not hasattr(config, "mlp_bias"):
    config.mlp_bias = False
if not hasattr(config, "attention_bias"):
    config.attention_bias = False

model = QTensorLlamaForCausalLM(config).to(torch.bfloat16)

ckpt_path = os.path.join(project_root, "integration", "qtensor_tinyllama_1.1b_healed.safetensors")
if os.path.exists(ckpt_path):
    print(f"Loading healed checkpoint from {ckpt_path}")
    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
else:
    print(f"Warning: {ckpt_path} not found. Running with unhealed random weights for UI testing.")

model.to(device)
model.eval()

def generate_text(prompt, max_tokens, temperature):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        do_sample=(temperature > 0.0),
    )
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    first_token_time = None
    
    for new_text in streamer:
        if first_token_time is None:
            first_token_time = time.time() - start_time
            
        generated_text += new_text
        ttft_str = f"{first_token_time*1000:.2f} ms"
        yield generated_text, "Streaming...", ttft_str, "Calculating..."
        
    end_time = time.time()
    total_time = end_time - start_time
    
    num_tokens = len(tokenizer.encode(generated_text))
    tok_sec = num_tokens / total_time if total_time > 0 else 0
    
    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    tok_sec_str = f"{tok_sec:.2f} tok/sec"
    vram_str = f"{vram_mb:.2f} MB"
    ttft_str = f"{first_token_time*1000:.2f} ms" if first_token_time else "N/A"
    
    yield generated_text, tok_sec_str, ttft_str, vram_str

# --- Gradio UI Layout ---
custom_css = """
body { background-color: #0b0f19; color: #ffffff; }
.gradio-container { border: 1px solid #1e293b; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
.metric-box { background: #1e293b; border-left: 4px solid #3b82f6; padding: 10px; border-radius: 8px; font-family: monospace; font-size: 1.2em; text-align: center; }
.metric-label { font-size: 0.8em; color: #94a3b8; text-transform: uppercase; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as demo:
    gr.Markdown("# 🚀 QTensor Edge Inference Engine: TinyLlama-1.1B")
    gr.Markdown("Zero-copy 1.58-bit INT8 MPO generation mapped directly to L1 SRAM via custom Triton Kernels.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Control Panel")
            prompt = gr.Textbox(label="System Prompt", lines=4, value="The capital of Australia is")
            max_tokens = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max Tokens")
            temperature = gr.Slider(minimum=0.0, maximum=1.5, value=0.7, step=0.1, label="Temperature")
            generate_btn = gr.Button("🚀 Ignite Generation", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### Generation Output")
            output_text = gr.Textbox(label="Real-time Stream", lines=8, interactive=False)
            
            gr.Markdown("### ⚡ Live Hardware Telemetry")
            with gr.Row():
                tok_sec = gr.Textbox(label="Inference Speed", value="-- tok/sec", interactive=False)
                ttft = gr.Textbox(label="Time to First Token (TTFT)", value="-- ms", interactive=False)
                vram_use = gr.Textbox(label="Peak VRAM Allocation", value="-- MB", interactive=False)
                
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt, max_tokens, temperature],
        outputs=[output_text, tok_sec, ttft, vram_use]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

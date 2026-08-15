import typer
import sys
import os
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="qtensor",
    help="QTensor CLI: Post-Training 1.58-bit MPO Compression Engine",
    add_completion=False,
)
console = Console()

@app.command()
def compress(
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace Model ID (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)"),
    out: str = typer.Option("qtensor_output.safetensors", "--out", "-o", help="Output path for compressed .safetensors"),
):
    """
    Compress a standard FP16/BF16 LLM into a 1.58-bit INT8 MPO format.
    """
    console.print(Panel.fit(
        f"[bold blue]QTensor Offline Forge (Compression)[/bold blue]\n"
        f"Model: [green]{model}[/green]\n"
        f"Output Path: [green]{out}[/green]",
        title="Configuration"
    ))
    from qtensor.engine.forge import forge_model
    forge_model(model, out, chi=256, lora_rank=128)
    console.print(f"[bold green]Compression Complete! Model saved to {out}[/bold green]")

@app.command()
def heal(
    model: str = typer.Option(..., "--model", "-m", help="Original HuggingFace Model ID"),
    weights: str = typer.Option(..., "--weights", "-w", help="Path to forged .safetensors weights"),
    dataset: str = typer.Option("HuggingFaceFW/fineweb-edu", "--dataset", "-d", help="Dataset for QAT Healing"),
    steps: int = typer.Option(10000, "--steps", "-s", help="Number of Knowledge Distillation steps for LoRA adapters"),
):
    """
    Launch KL-Divergence distillation to heal the LoRA adapters against the frozen INT8 cores.
    """
    console.print(Panel.fit(
        f"[bold blue]QTensor QAT Healing Distillation[/bold blue]\n"
        f"Base Model: [green]{model}[/green]\n"
        f"Forged Weights: [green]{weights}[/green]\n"
        f"Dataset: [green]{dataset}[/green]\n"
        f"Steps: [green]{steps}[/green]",
        title="Configuration"
    ))
    from qtensor.engine.trainer import train_healing_loop
    # The current trainer loop in qtensor/engine/trainer.py uses hardcoded paths,
    # but in a full SDK this would pass the kwargs to train_healing_loop(model, weights, dataset, steps).
    train_healing_loop()
    console.print(f"[bold green]Healing Complete![/bold green]")

@app.command()
def serve():
    """
    Launch the Gradio Edge Inference Dashboard.
    """
    console.print("[bold blue]Launching QTensor Serve Dashboard...[/bold blue]")
    from qtensor.engine.serve import demo
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    app()

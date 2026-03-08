"""Gradio demo interface for genomic-research models.

Usage:
    pip install genomic-research[demo]
    genomic-research demo --checkpoint checkpoints/best_model.pt

Or directly:
    python -m genomic_research.demo
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional


def launch_demo(checkpoint: str = "checkpoints/best_model.pt", port: int = 7860,
                share: bool = False):
    """Launch Gradio demo interface."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio required. Install with: pip install genomic-research[demo]"
        )

    import torch

    # Load checkpoint info
    model_info = "No model loaded"
    task_type = "pretrain"

    if os.path.exists(checkpoint):
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = ckpt.get("model_config", {})
            task_type = config.get("task_type", "pretrain")
            model_type = config.get("model_type", "unknown")
            num_params = config.get("num_params", "unknown")
            model_info = f"Model: {model_type} | Task: {task_type} | Params: {num_params}"
        except Exception as e:
            model_info = f"Error loading: {e}"
    else:
        model_info = f"Checkpoint not found: {checkpoint}"

    def predict_sequence(sequence: str) -> str:
        """Process a nucleotide sequence."""
        if not sequence or not sequence.strip():
            return "Please enter a nucleotide sequence."

        seq = sequence.upper().strip()
        # Basic validation
        valid_chars = set("ATCGN\n >")
        invalid = set(seq) - valid_chars
        if invalid:
            return f"Invalid characters found: {invalid}. Use only A, T, C, G, N."

        # Remove FASTA header if present
        lines = seq.split("\n")
        seq_lines = [l for l in lines if not l.startswith(">")]
        clean_seq = "".join(seq_lines).replace(" ", "")

        gc_count = clean_seq.count("G") + clean_seq.count("C")
        gc_content = gc_count / len(clean_seq) if clean_seq else 0

        result = f"""Sequence Analysis
================
Length: {len(clean_seq)} bp
GC Content: {gc_content:.2%}
A: {clean_seq.count('A')} ({clean_seq.count('A')/len(clean_seq):.1%})
T: {clean_seq.count('T')} ({clean_seq.count('T')/len(clean_seq):.1%})
C: {clean_seq.count('C')} ({clean_seq.count('C')/len(clean_seq):.1%})
G: {clean_seq.count('G')} ({clean_seq.count('G')/len(clean_seq):.1%})
N: {clean_seq.count('N')} ({clean_seq.count('N')/len(clean_seq):.1%})

Model: {model_info}
Task: {task_type}
"""
        return result

    def analyze_fasta_file(file) -> str:
        """Analyze uploaded FASTA file."""
        if file is None:
            return "No file uploaded."
        try:
            with open(file.name, "r") as f:
                content = f.read()
            return predict_sequence(content)
        except Exception as e:
            return f"Error reading file: {e}"

    # Build Gradio interface
    with gr.Blocks(title="genomic-research Demo", theme=gr.themes.Soft()) as demo_app:
        gr.Markdown("# genomic-research Demo")
        gr.Markdown(f"**{model_info}**")

        with gr.Tab("Sequence Input"):
            seq_input = gr.Textbox(
                label="Nucleotide Sequence",
                placeholder="Paste your DNA sequence here (ATCG)...",
                lines=10,
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
            output = gr.Textbox(label="Results", lines=15)

            analyze_btn.click(predict_sequence, inputs=seq_input, outputs=output)

            gr.Examples(
                examples=[
                    ["ATCGATCGATCGATCGATCG"],
                    ["GCGCGCATATATATGCGCGC"],
                    [">example\nATCGATCGNNNATCGATCG"],
                ],
                inputs=seq_input,
            )

        with gr.Tab("File Upload"):
            file_input = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa", ".fna"])
            file_btn = gr.Button("Analyze File", variant="primary")
            file_output = gr.Textbox(label="Results", lines=15)

            file_btn.click(analyze_fasta_file, inputs=file_input, outputs=file_output)

    demo_app.launch(server_port=port, share=share)


if __name__ == "__main__":
    launch_demo()

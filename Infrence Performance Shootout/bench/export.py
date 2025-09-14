"""Export Hugging Face models to ONNX format."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
except Exception:  # pragma: no cover - torch may not be installed
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore


def export_gpt2(path: Path, fp16: bool) -> None:
    if torch is None or AutoModelForCausalLM is None:
        raise RuntimeError("PyTorch/transformers not installed")
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    model.eval()
    if fp16:
        model.half()
    dummy = torch.zeros(1, 1, dtype=torch.long)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
    )


def export_bert(path: Path, fp16: bool) -> None:
    if torch is None or AutoModelForSequenceClassification is None:
        raise RuntimeError("PyTorch/transformers not installed")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    model.eval()
    if fp16:
        model.half()
    dummy = {
        "input_ids": torch.zeros(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
    }
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HF models to ONNX")
    parser.add_argument("--model", choices=["gpt2-medium", "bert-base-uncased"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.model.startswith("gpt2"):
        export_gpt2(args.out, args.fp16)
    else:
        export_bert(args.out, args.fp16)


if __name__ == "__main__":  # pragma: no cover
    main()

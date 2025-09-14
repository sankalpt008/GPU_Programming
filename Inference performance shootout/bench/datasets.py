"""Data loading utilities for the benchmarks."""
from __future__ import annotations

import itertools
from typing import Iterable, List

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional deps
    load_dataset = None  # type: ignore
    AutoTokenizer = None  # type: ignore


def synthetic_gpt2_prompts(seq_len: int, batch: int) -> List[List[int]]:
    """Return ``batch`` prompts each ``seq_len`` tokens long.

    Tokens are simply ascending integers modulo the vocab size (assumed 50257).
    """
    vocab = 50257
    prompts = []
    for b in range(batch):
        base = b * seq_len
        prompts.append([(base + i) % vocab for i in range(seq_len)])
    return prompts


def load_sst2(seq_len: int, batch: int) -> Iterable[dict]:
    """Yield tokenized SST-2 samples in mini-batches.

    When ``datasets`` or ``transformers`` is not available, synthetic sequences
    are generated instead.
    """
    if load_dataset is None or AutoTokenizer is None:  # pragma: no cover
        for _ in range(batch):
            yield {"input_ids": [0] * seq_len, "attention_mask": [1] * seq_len}
        return

    dataset = load_dataset("glue", "sst2", split="validation")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    iterator = iter(dataset)
    while True:
        batch_elems = list(itertools.islice(iterator, batch))
        if not batch_elems:
            break
        texts = [ex["sentence"] for ex in batch_elems]
        toks = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len)
        yield {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}

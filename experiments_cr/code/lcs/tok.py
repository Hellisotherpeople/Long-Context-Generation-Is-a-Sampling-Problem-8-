"""Shared tokenizer loader (applies known fixes)."""
from __future__ import annotations


def load_tokenizer(path):
    from transformers import AutoTokenizer
    if "istral" in path:
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True, fix_mistral_regex=True)
        except TypeError:
            pass
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)

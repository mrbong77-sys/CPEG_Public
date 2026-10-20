"""Gate1's two piece classifiers: a DeBERTa encoder with one linear head on the [CLS] vector, scored in fp32.

- piece classifier (checkpoint `piece_classifier.pt`): microsoft/deberta-v3-large, max 256 tokens (head + tail of a
  longer text), sigmoid score, then Platt-calibrated (config/gate1.json "piece_calibration");
- substance classifier S (checkpoint `substance_classifier.pt`): microsoft/deberta-v3-base, max 256 tokens (head +
  tail), sigmoid score, uncalibrated.

A checkpoint is a torch file {"backbone": state_dict, "head": state_dict} saved after fine-tuning, or the same
weights as one safetensors file with the keys "backbone.*" and "head.*" (the form in the model repository).
"""
from __future__ import annotations

import numpy as np

PIECE_ENC = "microsoft/deberta-v3-large"
SUBSTANCE_ENC = "microsoft/deberta-v3-base"
MAX_LEN, HEAD_TAIL, POOL, AMP = 256, True, "cls", False                      # both classifiers score with these settings


def hf(name: str, ckpt: bool = False):
    """(backbone, tokenizer, hidden size) for a Hugging Face encoder; ckpt turns on gradient checkpointing (memory only)."""
    from transformers import AutoModel, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(name)
    bb = AutoModel.from_pretrained(name)
    bb = bb.float()                                                           # fp32 encoder (the checkpoint's own dtype may be fp16)
    if ckpt:
        bb.gradient_checkpointing_enable()
    return bb, tk, bb.config.hidden_size


def encode(tok, batch: list[str], max_len: int = MAX_LEN, head_tail: bool = HEAD_TAIL) -> dict:
    """Tokenise to at most max_len tokens; with head_tail, a longer text keeps its first and last tokens."""
    if not head_tail:
        return tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    head = max_len // 2
    ids = [x if len(x) <= max_len else x[:head] + x[len(x) - (max_len - head):] for x in tok(batch)["input_ids"]]
    return tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")


def pool(hs, att, how: str = POOL):
    """Sentence vector: the [CLS] position (used here), the last attended position, or the masked mean."""
    import torch
    if how == "last":
        return hs[torch.arange(hs.shape[0], device=hs.device), att.sum(1) - 1]
    if how == "mean":
        m = att.unsqueeze(-1).to(hs.dtype)
        return (hs * m).sum(1) / m.sum(1).clamp(min=1)
    return hs[:, 0]


def load_state(path) -> dict:
    """{"backbone": state_dict, "head": state_dict} from a torch file or from a safetensors file (backbone.* / head.*)."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        flat = load_file(str(path))
        return {part: {k[len(part) + 1:]: v for k, v in flat.items() if k.startswith(part + ".")} for part in ("backbone", "head")}
    import torch
    return torch.load(path, map_location="cpu")


def predict(texts: list[str], encoder_name: str, path, log=print) -> np.ndarray:
    """Sigmoid scores of a saved classifier {"backbone", "head"} in the input order (batches of 64, fp32)."""
    import torch
    backbone, tok, hidden = hf(encoder_name)
    head = torch.nn.Linear(hidden, 1)
    st = load_state(path)
    backbone.load_state_dict(st["backbone"])
    head.load_state_dict(st["head"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(dev).eval()
    head.to(dev)
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = encode(tok, texts[i:i + 64])
            enc = {k: v.to(dev) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=AMP and dev.type == "cuda"):
                hs = backbone(**enc).last_hidden_state
            h = pool(hs, enc["attention_mask"])
            probs.append(torch.sigmoid(head(h.float()).squeeze(-1)).float().cpu().numpy())
    del backbone, head
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(probs) if probs else np.zeros(0)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(ab: dict, s) -> np.ndarray:
    """Platt calibration of the piece classifier: sigmoid(a * logit(s) + b), (a, b) fitted once on its validation split."""
    return 1 / (1 + np.exp(-(ab["a"] * _logit(s) + ab["b"])))


def fit_platt(y, s) -> dict:
    """The calibration fit: logistic regression on the logit of the raw score (C 1e6), rounded to 4 decimals."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(s).reshape(-1, 1), np.asarray(y, int))
    return {"a": round(float(m.coef_[0, 0]), 4), "b": round(float(m.intercept_[0]), 4)}

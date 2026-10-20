"""Untrained-unit filter: each benchmark text against the energy gate's training pool — same text (after strip), or
all-mpnet-base-v2 cosine >= 0.95 (near-duplicate). The training pool texts are not distributed (see README).
"""
from __future__ import annotations

import numpy as np

EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
NEAR = 0.95
SEED = 42


def embed(texts: list[str], log=print, batch: int = 64) -> np.ndarray:
    """Mean-pooled, L2-normalised all-mpnet-base-v2 vectors (fp32, deterministic, max 384 tokens)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    m = AutoModel.from_pretrained(EMB_MODEL).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=384, return_tensors="pt").to(dev)
            h = m(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(v, dim=1).cpu().numpy())
            if i // batch % 50 == 0:
                log(f"  embed {i + len(b['input_ids'])}/{len(texts)}")
    return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float32)


def overlap_flags(rows: list[dict], pool_texts: list[str], embed_fn=None, log=print) -> list[dict]:
    """Per item: {"id", "exact", "near", "cos"}; trained = exact or near."""
    ptxt = {t.strip() for t in pool_texts}
    embed_fn = embed_fn or (lambda ts: embed(ts, log))
    P = embed_fn(list(pool_texts))
    Q = embed_fn([r["text"] for r in rows])
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
    best = np.zeros(len(Q))
    for i in range(0, len(Q), 2048):
        best[i:i + 2048] = (Q[i:i + 2048] @ P.T).max(1)
    return [{"id": r["id"], "exact": r["text"].strip() in ptxt, "near": bool(c >= NEAR), "cos": round(float(c), 4)}
            for r, c in zip(rows, best)]

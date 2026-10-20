"""The energy-gate encoder: DeBERTa-v3-large fine-tuned as a one-logit sequence classifier (harmful = 1). Its logit
of a text is the text's energy score s; the six scores of a request and its five probes feed the logistic head.

Fine-tuning recipe of the CPEG encoder: max 512 tokens, lr 1e-5, batch 8 x 4 accumulation, 2 epochs, warm-up 6%,
AdamW (weight decay 0.01), bf16 autocast on fp32 weights, each unique training row once with its loss weighted by its
repeat weight and label balance (row_weights), length-grouped batches, seed 42.

Deterministic: seed 42, one fixed row order, greedy scoring in fp32.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

SEED = 42
MODEL, MAX_LEN, LR, BATCH, ACCUM, EPOCHS, WARMUP = "microsoft/deberta-v3-large", 512, 1e-5, 8, 4, 1, 0.06
FINAL_EPOCHS = 2                                                              # the CPEG encoder: train(..., epochs=2)
GROUP = 128                                                                   # batches per length-sorted group
RESUME_EVERY = 2000                                                           # optimiser steps per resume checkpoint
CHECK = {"model": "microsoft/deberta-v3-small", "max_len": 64, "rows": 64}


def row_weights(y: np.ndarray, rep: np.ndarray) -> np.ndarray:
    """Each unique row once, its loss weighted by `rep`, the two labels scaled to the same total weight (the
    repeated and class-balanced training set in expectation)."""
    w = rep.astype(np.float64)
    pos, neg = w[y == 1].sum(), w[y == 0].sum()
    big = max(pos, neg)
    w = np.where(y == 1, w * big / max(pos, 1e-9), w * big / max(neg, 1e-9))
    return w / w.mean()


def batches(rows: np.ndarray, lengths: np.ndarray, batch: int, group: int = GROUP) -> list[np.ndarray]:
    """Length-grouped batches: the fixed row order is cut into groups of `group` batches, each group sorted by text
    length and cut into batches, the batch order inside a group shuffled with seed 42 (less padding, same rows)."""
    rng = np.random.RandomState(SEED)
    out = []
    span = batch * group
    for i in range(0, len(rows), span):
        g = rows[i:i + span]
        g = g[np.argsort(lengths[g], kind="stable")]
        bs = [g[j:j + batch] for j in range(0, len(g), batch)]
        out += [bs[k] for k in rng.permutation(len(bs))]
    return out


def _tok(tok, items: list[str], max_len: int, chunk: int = 10000) -> list:
    """Token ids (int32 arrays) of the texts, in the input order; tokenised in chunks to bound memory."""
    out = [None] * len(items)
    idx = list(range(len(items)))
    for c in range(0, len(idx), chunk):
        sub = idx[c:c + chunk]
        enc = tok([items[i] for i in sub], truncation=True, max_length=max_len)["input_ids"]
        for i, e in zip(sub, enc):
            out[i] = np.asarray(e, dtype=np.int32)
    return out


def _det():
    import random
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def train(texts: list, y: np.ndarray, rep: np.ndarray, out: Path, check: bool = False, log=print,
          epochs: int = EPOCHS, init: str | Path | None = None, model_name: str | None = None) -> Path:
    """Fine-tune once and save to `out` (kept if already there). `init`
    starts from a fine-tuned checkpoint; a checkpoint every RESUME_EVERY optimiser steps lets an interrupted run go on
    from there (same batch order; the dropout stream after a resume is not the uninterrupted one)."""
    import torch
    import transformers as T
    if (out / "done.json").exists():
        log(f"  fine-tuned encoder already at {out}")
        return out
    _det()
    torch.backends.cuda.matmul.allow_tf32 = True                              # fp32 weights and activations, TF32 math
    torch.backends.cudnn.allow_tf32 = True
    name, max_len = (CHECK["model"], CHECK["max_len"]) if check else (model_name or MODEL, MAX_LEN)
    src = str(init) if init else name
    wts = row_weights(y, rep)
    rows = np.random.RandomState(SEED).permutation(len(y))                    # every unique row once
    if check:
        rows = rows[: CHECK["rows"]]
    tok = T.AutoTokenizer.from_pretrained(src)
    model = T.AutoModelForSequenceClassification.from_pretrained(src, num_labels=1, dtype=torch.float32).to(
        "cuda" if torch.cuda.is_available() else "cpu")                      # transformers 5 keeps the checkpoint's fp16
    assert all(p.dtype == torch.float32 for p in model.parameters())          # otherwise: Adam on fp16 weights -> NaN
    dev = model.device                                                        # no gradient checkpointing unless needed
    ids = _tok(tok, texts, max_len)                                           # tokenised once, not per batch
    lengths = np.array([len(x) for x in ids])
    pad = tok.pad_token_id
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = math.ceil(len(rows) * epochs / (BATCH * ACCUM))
    sch = T.get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    model.train()
    yt, wt = torch.tensor(y, dtype=torch.float32), torch.tensor(wts, dtype=torch.float32)  # moved to the GPU once
    log(f"  fine-tune {name}: {len(rows)} unique rows, loss weighted by rep and label balance, {steps} optimiser steps,"
        f" max_len {max_len}")
    import time
    if dev.type == "cuda":
        # Some drivers spill to system memory instead of raising out-of-memory, so a batch that does not fit crawls. Cap the allocator so it raises, and keep room for
        # AdamW's two moment buffers (allocated at the first step, after this probe)
        total = torch.cuda.get_device_properties(dev).total_memory
        pbytes = sum(p.numel() * p.element_size() for p in model.parameters())
        torch.cuda.set_per_process_memory_fraction(max(0.05, min(0.95, (0.95 * total - 2 * pbytes) / total)), dev)
        log(f"  GPU {total / 2**30:.1f} GiB, weights {pbytes / 2**30:.1f} GiB; probe cap leaves "
            f"{2 * pbytes / 2**30:.1f} GiB for the optimiser")
    if dev.type == "cuda":                                                    # the longest batch once, no
        x = None
        try:                                                                  # step; checkpointing only if it won't fit
            x = torch.full((BATCH, int(lengths.max())), pad, dtype=torch.long, device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(input_ids=x).logits.float().sum().backward()
            log(f"  longest batch fits without checkpointing (peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
        except torch.OutOfMemoryError:
            import gc
            gc.collect()                                                      # the traceback held the activations
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            model.gradient_checkpointing_enable()
            log("  longest batch does not fit: gradient checkpointing on")
        model.zero_grad(set_to_none=True)
        del x
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95, dev)                # training: raise rather than page
    yt, wt = yt.to(dev), wt.to(dev)
    ok = torch.ones((), dtype=torch.bool, device=dev)
    t0, k, k0 = time.time(), 0, 0
    rf = out.parent / (out.name + "_resume.pt")
    if rf.exists():                                                           # go on from the last checkpoint
        st = torch.load(rf, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sch.load_state_dict(st["sch"])
        k0 = int(st["k"])
        log(f"  resumed at optimiser step {k0 // ACCUM}")
    for ep in range(epochs):
        order = rows if ep == 0 else np.random.RandomState(SEED + ep).permutation(rows)   # a new order each epoch
        for b in batches(order, lengths, BATCH):
            if k < k0:                                                        # already trained before the resume
                k += 1
                continue
            n = int(lengths[b].max())
            x = torch.full((len(b), n), pad, dtype=torch.long)
            for r, j in enumerate(b):
                x[r, :lengths[j]] = torch.from_numpy(ids[j].astype(np.int64))
            x = x.to(dev, non_blocking=True)
            bt = torch.as_tensor(b, device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit = model(input_ids=x, attention_mask=(x != pad).long()).logits.squeeze(-1).float()
            per = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[bt], reduction="none")
            loss = (per * wt[bt]).mean() / ACCUM                              # weights have mean 1 over the pool
            ok &= torch.isfinite(loss)                                        # no host sync per micro-batch
            if k % 50 == 49 and not bool(ok):                                 # stop within 50 micro-batches of a NaN
                raise FloatingPointError(f"non-finite loss by micro-batch {k + 1}")
            loss.backward()
            k += 1
            if k % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # last partial accumulation is dropped
                opt.step()
                sch.step()
                opt.zero_grad()
                if (k // ACCUM) % RESUME_EVERY == 0:
                    tmp = rf.with_suffix(".tmp")
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "k": k}, tmp)
                    os.replace(tmp, rf)
            if k % (ACCUM * 500) == 0 or k // ACCUM in (10, 100) and k % ACCUM == 0:
                log(f"  step {k // ACCUM} / {steps}, loss {float(loss) * ACCUM:.4f}, "
                    f"{(time.time() - t0) / max((k - k0) // ACCUM, 1):.2f} s/step")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "done.json").write_text(json.dumps({"model": name, "init": str(init) if init else None, "rows": int(len(rows)),
                                               "rows_mode": "unique, weighted",
                                               "steps": steps, "max_len": max_len,
                                               "lr": LR, "batch": BATCH * ACCUM, "epochs": epochs, "seed": SEED}), encoding="utf-8")
    if rf.exists():
        rf.unlink()
    del model
    torch.cuda.empty_cache()
    return out


_CACHE: dict = {}


def _ckey(t) -> str:
    import hashlib
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def score(ckpt: str | Path, texts: list[str], log=print, batch: int = 64, cache: bool = True) -> np.ndarray:
    """Logits of every text, in the input order. With `cache`, the checkpoint folder keeps a score cache
    (score_cache.json, sha1 of the text -> logit), so repeated texts are scored once (outputs are the same either way)."""
    if not texts:
        return np.zeros(0)
    if not cache:
        return _score(ckpt, texts, log, batch)
    cf = Path(ckpt) / "score_cache.json"
    have = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else {}
    keys = [_ckey(t) for t in texts]
    todo = sorted({k: t for k, t in zip(keys, texts) if k not in have}.items(), key=lambda kv: kv[0])
    if todo:
        new = _score(ckpt, [t for _, t in todo], log, batch)
        have.update({k: float(v) for (k, _), v in zip(todo, new)})
        tmp = cf.with_suffix(".tmp")
        tmp.write_text(json.dumps(have), encoding="utf-8")
        os.replace(tmp, cf)
    return np.array([have[k] for k in keys], dtype=np.float64)


def _score(ckpt: str | Path, texts: list[str], log=print, batch: int = 64) -> np.ndarray:
    """Logits of every text, longest first for speed, in the input order."""
    import torch
    import transformers as T
    key = str(ckpt)
    if key not in _CACHE:
        _CACHE.clear()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        info = json.loads((Path(ckpt) / "done.json").read_text(encoding="utf-8"))
        _CACHE[key] = (T.AutoTokenizer.from_pretrained(ckpt),
                       T.AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.float32).to(dev).eval(),
                       info["max_len"])
    tok, model, max_len = _CACHE[key]
    ids = _tok(tok, texts, max_len)
    order = np.argsort([-len(x) for x in ids], kind="stable")
    out = np.zeros(len(texts), dtype=np.float64)
    pad = tok.pad_token_id
    with torch.no_grad():
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            x = torch.full((len(b), max(len(ids[j]) for j in b)), pad, dtype=torch.long)
            for r, j in enumerate(b):
                x[r, :len(ids[j])] = torch.from_numpy(ids[j].astype(np.int64))
            x = x.to(model.device)
            out[b] = model(input_ids=x, attention_mask=(x != pad).long()).logits.squeeze(-1).float().cpu().numpy()
            if i // batch % 200 == 0:
                log(f"  ft-encoder score {i + len(b)} / {len(texts)}")
    return out

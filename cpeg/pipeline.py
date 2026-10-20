"""CPEG inference: Gate1 tier + energy E -> band routing -> block / delegate / pass, with an explanation record per request.

    from cpeg.pipeline import CPEG
    g = CPEG(piece_ckpt="ckpt/piece_classifier.pt", s_ckpt="ckpt/substance_classifier.pt", energy_ckpt="ckpt/energy_encoder")
    records = g.decide(["How do I kill a Python process?"])

    g = CPEG.from_pretrained("path/to/model-repo")          # or a Hugging Face model repository id (downloads it)

Each request: Gate1 parses and scores it (two classifiers, 25 features, boosted head, two thresholds); the energy gate
scores it and its five probes with the fine-tuned encoder and the logistic head gives E; band_tier combines the two.
Both stages run on every request. Texts are used as given (a multi-turn conversation: user turns joined by newlines).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .energy import head as EH
from .gate1.head import Gate1
from .route import band_tier, energy_only

CONFIG = Path(__file__).resolve().parents[1] / "config"


def load_no_probe_bands(config_dir: Path = CONFIG) -> dict:
    """Bands of the no-probes ablation (E = s0, the same encoder's logit of the request)."""
    op = json.loads((config_dir / "operating_point.json").read_text(encoding="utf-8"))
    return op["ablation_no_probes"]["bands"]


def load_energy_head(config_dir: Path = CONFIG) -> dict:
    """{"w": [w_text, w_malice_added, ..., w_protective], "b": b, "bands": {...}} from the committed configuration."""
    h = json.loads((config_dir / "energy_head.json").read_text(encoding="utf-8"))
    op = json.loads((config_dir / "operating_point.json").read_text(encoding="utf-8"))
    return {"w": [h["weights"][k] for k in h["feature_order"]], "b": h["intercept"], "bands": op["cpeg"]["bands"]}


class CPEG:
    def __init__(self, piece_ckpt, s_ckpt, energy_ckpt, config_dir: Path = CONFIG, gate1_thresholds: str = "registered",
                 score_cache: bool = True):
        self.gate1 = Gate1(piece_ckpt, s_ckpt, gate1_thresholds, config_dir)
        self.energy_ckpt = energy_ckpt
        self.score_cache = score_cache                                        # score_cache.json in the encoder folder
        self.mod = load_energy_head(config_dir)
        self.no_probe_bands = load_no_probe_bands(config_dir)

    @classmethod
    def from_pretrained(cls, path_or_id, revision: str | None = None, **kw) -> "CPEG":
        """The model-repository layout: piece_gate/{piece,substance}_classifier.safetensors and energy_encoder/. A path
        that is not a local folder is read as a Hugging Face repository id and downloaded (huggingface_hub). Nothing is
        written into the folder (no score cache)."""
        root = Path(path_or_id)
        if not root.is_dir():
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(str(path_or_id), revision=revision))
        files = {"piece_ckpt": root / "piece_gate" / "piece_classifier.safetensors",
                 "s_ckpt": root / "piece_gate" / "substance_classifier.safetensors", "energy_ckpt": root / "energy_encoder"}
        missing = [str(f) for f in files.values() if not f.exists()]
        if missing:
            raise FileNotFoundError(f"not a CPEG model folder, missing: {missing}")
        g = cls(**{k: str(v) for k, v in files.items()}, **{"score_cache": False, **kw})
        h = root / "heads" / "cpeg.json"                                      # full-precision head and bands of the release
        if h.exists():
            d = json.loads(h.read_text(encoding="utf-8"))
            g.mod = {"w": [float(x) for x in d["w"]], "b": float(d["b"]), "bands": {k: float(v) for k, v in d["bands"].items()}}
        return g

    def decide(self, texts: list[str], log=print) -> list[dict]:
        p, tiers0 = self.gate1.tiers(texts, log)
        Z = EH.cfx_features(self.energy_ckpt, texts, log, self.score_cache)
        rec = EH.explain(self.mod, Z, list(tiers0))
        for r, pi in zip(rec, p):
            r["gate1_p"] = round(float(pi), 6)
        return rec

    def tiers(self, texts: list[str], log=print) -> list[str]:
        return [r["decision"] for r in self.decide(texts, log)]

    def energy(self, texts: list[str], log=print) -> np.ndarray:
        return EH.cfx_energy(self.mod, EH.cfx_features(self.energy_ckpt, texts, log, self.score_cache))

    def energy_gate_only_tiers(self, texts: list[str], log=print) -> list[str]:
        """Ablation "energy gate only": no piece gate (E >= theta_block blocks, every other request passes)."""
        return [energy_only(float(x), self.mod["bands"]) for x in self.energy(texts, log)]

    def no_probes_tiers(self, texts: list[str], log=print) -> list[str]:
        """Ablation "no probes": E = s0 (the same encoder's logit of the request) with its own bands."""
        _, tiers0 = self.gate1.tiers(texts, log)
        s0 = EH.ftenc.score(self.energy_ckpt, texts, log, cache=self.score_cache)  # = cfx_features(...)[:, 0]
        return route_from(list(tiers0), s0, self.no_probe_bands)


def route_from(tiers0: list[str], E: np.ndarray, bands: dict) -> list[str]:
    """Band routing from precomputed Gate1 tiers and energies (e.g. to apply the no-probes bands to the text logit s0)."""
    return [band_tier(t, float(x), bands) for t, x in zip(tiers0, E)]

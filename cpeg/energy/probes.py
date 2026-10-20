"""The five rule-based counterfactual probes (no LLM, deterministic).

Each request gets five rewrites; the energy head reads the encoder logit of the request together with how the logit
moves under each rewrite. A request whose meaning hangs on intent moves a lot when malice is added or trigger words are
masked; a request that is harmful in itself barely moves.

Order (cf_variants): original, + malicious intent, trigger words masked, + harmless reason, people swapped for an
inanimate target, protective reframing.
"""
from __future__ import annotations

import re

CF_MAL = " I want to use this to seriously hurt a specific person without getting caught."
CF_BEN = "I am asking for a legitimate and harmless reason: "
CF_LEX = ("kill", "killing", "murder", "bomb", "bombs", "explosive", "explosives", "weapon", "weapons", "gun", "guns",
          "poison", "poisoning", "drug", "drugs", "meth", "cocaine", "heroin", "hack", "hacking", "hacker", "malware",
          "virus", "steal", "stealing", "theft", "fraud", "scam", "attack", "attacks", "shoot", "shooting", "stab",
          "terror", "terrorist", "terrorism", "suicide", "die", "death", "dead", "hurt", "harm", "abuse", "rape",
          "kidnap", "smuggle", "smuggling", "launder", "laundering", "counterfeit", "exploit", "crack", "bypass",
          "illegal", "blood", "knife", "burn", "destroy", "nuclear", "chemical", "weaponize", "torture", "violence",
          "violent", "assault", "arson", "trafficking", "blackmail", "extort", "ransomware", "phishing")
CF_PEOPLE = ("person", "people", "someone", "somebody", "him", "her", "them", "man", "woman", "men", "women", "child",
             "children", "kid", "kids", "wife", "husband", "neighbor", "neighbour", "boss", "girlfriend", "boyfriend",
             "friend", "teacher", "student", "family", "baby", "victim", "coworker", "colleague")
CF_PROTECT = "How can people protect themselves against the following: "
CF_VARIANTS = 6                                                               # the final system uses all six (original + 5 probes)
CFX_PROBES = ("malice_added", "trigger_masked", "benign_reason", "object_swap", "protective")   # cf_variants order


def _cf_mask(t: str) -> str:
    """Trigger words (CF_LEX, whole words, any case) replaced by 'something'."""
    return re.sub(r"\b(" + "|".join(CF_LEX) + r")\b", "something", t, flags=re.I)


def _cf_object(t: str) -> str:
    """Words for people (CF_PEOPLE, whole words, any case) replaced by 'a computer program'."""
    return re.sub(r"\b(" + "|".join(CF_PEOPLE) + r")\b", "a computer program", t, flags=re.I)


def cf_variants(t: str, n: int | None = None) -> list[str]:
    """Original, + malicious intent, trigger words masked, + harmless reason; with n >= 6 also people swapped for an
    inanimate target, and the protective reframing."""
    v = [t, t + CF_MAL, _cf_mask(t), CF_BEN + t]
    if (n or CF_VARIANTS) >= 6:
        v += [_cf_object(t), CF_PROTECT + t]
    return v

# utils/helper.py
from __future__ import annotations
import json

import unicodedata
import emoji


def b(x, default=False):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ("true","1","y","yes"): return True
        if v in ("false","0","n","no"): return False
    return default

def llist(x):
    return x if isinstance(x, list) else []

def normalize_gpt_json(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        parts = s.split("\n", 1)
        if parts and parts[0].lower().startswith("json"):
            s = parts[1] if len(parts) > 1 else ""
    if "{" in s and "}" in s:
        s = s[s.find("{"): s.rfind("}") + 1]
    try:
        return json.loads(s)
    except Exception:
        return {}

def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    return unicodedata.normalize("NFC", s)

def has_emoji(text: str) -> bool:
    return any(ch in emoji.EMOJI_DATA for ch in text or "")

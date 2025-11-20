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

# def normalize_gpt_json(raw):
#     if isinstance(raw, dict):
#         return raw
#     if not isinstance(raw, str):
#         return {}
#     s = raw.strip()
#     if s.startswith("```"):
#         s = s.strip("`")
#         parts = s.split("\n", 1)
#         if parts and parts[0].lower().startswith("json"):
#             s = parts[1] if len(parts) > 1 else ""
#     if "{" in s and "}" in s:
#         s = s[s.find("{"): s.rfind("}") + 1]
#     try:
#         return json.loads(s)
#     except Exception:
#         return {}

def normalize_gpt_json_cat(raw):
    if isinstance(raw, list):
        return [str(x).strip() for x in raw]
    if isinstance(raw, dict):
        return list(map(str, raw.keys()))

    if isinstance(raw, str):
        parsed = normalize_gpt_json(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
        if isinstance(parsed, dict):
            return list(map(str, parsed.keys()))
    return []

import re, json

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
        
    s = re.sub(r'\]\s*\.join\(\s*(?:\'|").*?(?:\'|")\s*\)', ']', s)

    # --- 기존 1차 시도 ---
    try:
        return json.loads(s)
    
    except Exception:
        pass
    
    def _fix_bracket_mismatch(text: str) -> str:
        out = []
        stack = [] 
        in_string = False
        escape = False

        for ch in text:
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                else:
                    if ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                continue
            if ch == '"':
                in_string = True
                out.append(ch)
                continue

            if ch == '{':
                stack.append('}')
                out.append(ch)
                continue

            if ch == '[':
                stack.append(']')
                out.append(ch)
                continue

            if ch == '}' or ch == ']':
                if stack:
                    expected = stack[-1]
                    if ch == expected:
                        stack.pop()
                        out.append(ch)
                    else:
                        out.append(expected)
                        stack.pop()
                else:
                    out.append(ch)
                continue

            out.append(ch)

        while stack:
            out.append(stack.pop())

        return "".join(out)

    s_fixed_brackets = _fix_bracket_mismatch(s)
    try:
        return json.loads(s_fixed_brackets)
    except Exception:
        pass

    def _fix_unescaped_quotes(text: str) -> str:
        out = []
        in_string = False
        escape = False
        i, n = 0, len(text)

        def next_meaningful(idx: int) -> str:
            j = idx
            while j < n and text[j] in (" ", "\t", "\r", "\n"):
                j += 1
            return text[j] if j < n else ""

        while i < n:
            ch = text[i]
            if not in_string:
                if ch == '"':
                    in_string = True
                    escape = False
                    out.append(ch)
                else:
                    out.append(ch)
                i += 1
                continue

            if escape:
                out.append(ch)
                escape = False
                i += 1
                continue

            if ch == '\\':
                out.append(ch)
                escape = True
                i += 1
                continue

            if ch == '"':
                nm = next_meaningful(i + 1)
                if nm in (",", "]", "}", ""):
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
                i += 1
                continue

            out.append(ch)
            i += 1

        return "".join(out)
    try:
        fixed = _fix_unescaped_quotes(s_fixed_brackets)
        fixed = _fix_bracket_mismatch(fixed)
        return json.loads(fixed)
    except Exception:
        return {}



def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    return unicodedata.normalize("NFC", s)

def has_emoji(text: str) -> bool:
    return any(ch in emoji.EMOJI_DATA for ch in text or "")

# utils/gpt_client.py — async clients with semaphore, timeout, deterministic backoff (no console prints)
import os
import json
import asyncio
from typing import List, Tuple

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# ====== async control knobs ======
# _ASYNC_TIMEOUT_SEC = 45
# _ASYNC_MAX_RETRIES = 4
# _SEMAPHORE = asyncio.Semaphore(24)

_ASYNC_TIMEOUT_SEC = 3600        # 사실상 무제한 (1시간)
_ASYNC_MAX_RETRIES = 10          # 최대 재시도 횟수 확대
_SEMAPHORE = asyncio.Semaphore(1)  # 동시성 1개 → 가장 안전

def set_async_limits(timeout_sec: int = 45, max_retries: int = 4, concurrency: int | None = None):
    """
    Configure timeout/retries/concurrency for async chat calls.
    """
    global _ASYNC_TIMEOUT_SEC, _ASYNC_MAX_RETRIES, _SEMAPHORE
    _ASYNC_TIMEOUT_SEC = timeout_sec
    _ASYNC_MAX_RETRIES = max_retries
    if concurrency is not None:
        _SEMAPHORE = asyncio.Semaphore(concurrency)

def ask_gpt4o(messages: List[dict], model="gpt-4o", temperature=0.0) -> Tuple[str | list, dict]:
    """
    Sync wrapper (legacy). Returns (reply, usage-like dict) or ("error", {} on failure).
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        reply = response["choices"][0]["message"]["content"].strip()
        usage = response.get("usage", {})
        if reply.startswith("["):
            try:
                return json.loads(reply), usage
            except json.JSONDecodeError:
                return [], usage
        return reply, usage
    except Exception:
        return "error", {}

def ask_gpt5(messages: List[dict], model="gpt-5") -> Tuple[str | list, dict]:
    """
    Sync wrapper (legacy). Returns (reply, usage-like dict) or ("error", {}).
    """
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages)
        reply = response["choices"][0]["message"]["content"].strip()
        usage = response.get("usage", {})
        if reply.startswith("["):
            try:
                return json.loads(reply), usage
            except json.JSONDecodeError:
                return [], usage
        return reply, usage
    except Exception:
        return "error", {}

async def _chat_acreate_with_retry(model: str, messages: List[dict], *, temperature: float | None = None) -> Tuple[str | list, dict]:
    """
    Async wrapper with semaphore + timeout + deterministic exponential backoff.
    On final failure returns ("error", {}).
    """
    base_backoff = 0.6
    for attempt in range(_ASYNC_MAX_RETRIES):
        try:
            async with _SEMAPHORE:
                kwargs = dict(model=model, messages=messages)
                if temperature is not None:
                    kwargs["temperature"] = temperature
                resp = await asyncio.wait_for(
                    openai.ChatCompletion.acreate(**kwargs),
                    timeout=_ASYNC_TIMEOUT_SEC
                )
            reply = resp["choices"][0]["message"]["content"].strip()
            usage = resp.get("usage", {})
            if reply.startswith("["):
                try:
                    return json.loads(reply), usage
                except json.JSONDecodeError:
                    return [], usage
            return reply, usage
        except Exception:
            if attempt < _ASYNC_MAX_RETRIES - 1:
                await asyncio.sleep(base_backoff * (2 ** attempt))
            else:
                return "error", {}

async def ask_gpt4o_async(messages: List[dict], model="gpt-4o", timeout: int | None = None, max_retries: int | None = None):
    """
    Async GPT-4o call with configured limits; returns (reply, usage-like dict) or ("error", {}).
    """
    if timeout is not None:
        set_async_limits(timeout_sec=timeout, max_retries=max_retries or _ASYNC_MAX_RETRIES)
    return await _chat_acreate_with_retry(model, messages, temperature=0.0)

async def ask_gpt5_async(messages: List[dict], model="gpt-5", timeout: int | None = None, max_retries: int | None = None):
    """
    Async GPT-5 call with configured limits; returns (reply, usage-like dict) or ("error", {}).
    """
    if timeout is not None:
        set_async_limits(timeout_sec=timeout, max_retries=max_retries or _ASYNC_MAX_RETRIES)
    return await _chat_acreate_with_retry(model, messages)

# graph/line_subgraph.py — line-level LangGraph with JSONL error logging
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
import os, json, time
from threading import Lock

from utils.gpt_client import ask_gpt4o_async, ask_gpt5_async
from prompt_builder.build_prompt import (
    build_category_prompt,
    build_check_prompt,
    build_emoji_check_prompt,
)
from utils.helper import b, llist, normalize_gpt_json, norm, has_emoji

_ERROR_LOG_LOCK = Lock()

def _error_log_path(output_dir: Optional[str]) -> str:
    base = output_dir or os.getenv("OUTPUT_DIR") or os.getcwd()
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "error.jsonl")

def _append_error_jsonl(payload: dict, output_dir: Optional[str]) -> None:
    path = _error_log_path(output_dir)
    line = json.dumps(payload, ensure_ascii=False)
    with _ERROR_LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def _log_error_line(
    state_like: Dict[str, Any],
    *,
    stage: str,
    line_no: Optional[int],
    category: Optional[str],
    error_type: str,
    error_message: str,
) -> None:
    """
    Append a line-level error as one JSON line into error.jsonl
    """
    payload = {
        "type": "gpt_call_error",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "context": {
            "parent_folder": state_like.get("parent_folder", "unknown"),
            "filename": state_like.get("filename", "unknown"),
        },
        "stage": stage,
        "line_no": line_no,
        "category": category,
        "error": {"type": error_type, "message": error_message},
        "guideline": None,
    }
    _append_error_jsonl(payload, state_like.get("output_dir"))

async def safe_ask(
    func,
    messages,
    *,
    model,
    timeout,
    max_retries,
    stage: str,
    state_for_log: Optional[Dict[str, Any]] = None,
    line_no: Optional[int] = None,
    category: Optional[str] = None,
):
    """
    Wrapper for GPT calls with JSONL error logging (line-level)
    """
    try:
        return await func(messages, model=model, timeout=timeout, max_retries=max_retries)
    except Exception as e:
        if state_for_log is not None:
            _log_error_line(
                state_for_log,
                stage=stage,
                line_no=line_no,
                category=category,
                error_type=type(e).__name__,
                error_message=str(e),
            )
        return "error", {}


class LineState(TypedDict, total=False):
    # input
    i: int
    src_line: str
    trn_line: str
    target: str

    # optional meta for error logging
    parent_folder: str
    filename: str
    output_dir: str

    # work
    revised_fmt: str
    detected_categories: List[str]
    violated_categories: List[str]
    spans_by_category: Dict[str, Dict[str, List[Any]]]
    emoji_issue_item: Optional[Dict[str, Any]]
    checked_sentence_item: Optional[Dict[str, Any]]

    # failure logs (in-memory markers if needed)
    failures: List[str]


class DetectCategoryNode:
    def __init__(self, api_timeout: int, max_retries: int):
        self.timeout = api_timeout
        self.max_retries = max_retries

    async def __call__(self, state: LineState) -> LineState:
        s = state.copy()
        s.setdefault("failures", [])
        s["revised_fmt"] = s.get("trn_line", "")

        # 숫자가 하나도 없다면 카테고리 검출 생략
        if not any(ch.isdigit() for ch in (s["src_line"] or "")) and not any(ch.isdigit() for ch in (s["trn_line"] or "")):
            s["detected_categories"] = []
            return s

        sys_cat, usr_cat = build_category_prompt(s["revised_fmt"])
        res, _ = await safe_ask(
            ask_gpt4o_async, [sys_cat, usr_cat],
            model='gpt-4o',
            timeout=self.timeout, max_retries=self.max_retries,
            stage="category",
            state_for_log=s,
            line_no=(s.get("i", -1) + 1),
            category=None,
        )
        cats: List[str] = []
        if res != "error":
            if isinstance(res, str):
                cats = [res]
            elif isinstance(res, list):
                cats = [c for c in res if isinstance(c, str)]
        s["detected_categories"] = cats
        s["violated_categories"] = []
        s["spans_by_category"] = {}
        return s


class FormatCheckLoopNode:
    def __init__(self, api_timeout: int, max_retries: int, get_guideline):
        self.timeout = api_timeout
        self.max_retries = max_retries
        self.get_guideline = get_guideline

    async def __call__(self, state: LineState) -> LineState:
        s = state.copy()
        cats = list(dict.fromkeys(s.get("detected_categories") or []))  # unique & stable
        for cat in cats:
            guideline = self.get_guideline(s["target"], cat)
            if not guideline:
                continue
            before = s["revised_fmt"]
            sys_chk, usr_chk = build_check_prompt(before, guideline, s["src_line"])
            res, _ = await safe_ask(
                ask_gpt5_async, [sys_chk, usr_chk],
                model='gpt-5',
                timeout=self.timeout, max_retries=self.max_retries,
                stage="format_check",
                state_for_log=s,
                line_no=(s.get("i", -1) + 1),
                category=cat,
            )
            tmp_src_sp, tmp_trn_sp, tmp_rev_sp = [], [], []
            if res != "error":
                js = normalize_gpt_json(res)
                if js:
                    new_rev = js.get("revised", before)
                    if isinstance(new_rev, str):
                        s["revised_fmt"] = new_rev.strip()
                    tmp_src_sp = llist(js.get("source_spans"))
                    tmp_trn_sp = llist(js.get("trans_spans"))
                    tmp_rev_sp = llist(js.get("revised_spans"))
                elif isinstance(res, str):
                    s["revised_fmt"] = res.strip()

            if norm(s["revised_fmt"]) != norm(before):
                s["violated_categories"].append(cat)
                if tmp_src_sp or tmp_trn_sp or tmp_rev_sp:
                    s["spans_by_category"][cat] = {
                        "source_spans": tmp_src_sp,
                        "trans_spans": tmp_trn_sp,
                        "revised_spans": tmp_rev_sp
                    }
        return s


class EmojiCheckNode:
    def __init__(self, api_timeout: int, max_retries: int):
        self.timeout = api_timeout
        self.max_retries = max_retries

    async def __call__(self, state: LineState) -> LineState:
        s = state.copy()
        src = s["src_line"]; cur = s["revised_fmt"]
        if not (has_emoji(src) or has_emoji(cur)):
            return s

        sys1, usr1 = build_emoji_check_prompt(src, cur)
        res, _ = await safe_ask(
            ask_gpt5_async, [sys1, usr1],
            model='gpt-5',
            timeout=self.timeout, max_retries=self.max_retries,
            stage="emoji_check",
            state_for_log=s,
            line_no=(s.get("i", -1) + 1),
            category=None,
        )
        js = normalize_gpt_json(res) if res != "error" else {}
        suggestion = None
        sugs = js.get("suggestions")
        if isinstance(sugs, list):
            for v in sugs:
                if isinstance(v, str) and v.strip():
                    suggestion = v.strip(); break
        emoji_issue = b(js.get("emoji_issue"), False)
        if emoji_issue or (suggestion and suggestion != cur):
            s["emoji_issue_item"] = {
                "line_no": s["i"] + 1,
                "source_line": src,
                "trans_line": cur,
                "suggestion": suggestion or cur
            }
            s["revised_fmt"] = suggestion or cur
        return s


class LineReduceNode:
    async def __call__(self, state: LineState) -> LineState:
        s = state.copy()
        det = s.get("detected_categories", [])
        det = list(dict.fromkeys(det)) 

        if det:
            s["checked_sentence_item"] = {
                "detected_categories": det,
                "violated_categories": s.get("violated_categories", []),
                "spans_by_category": s.get("spans_by_category", {})
            }
        return s



def build_line_subgraph(api_timeout: int, max_retries: int, get_guideline):
    """Build and return compiled line-level LangGraph"""
    g = StateGraph(LineState)
    g.add_node("detect_category", DetectCategoryNode(api_timeout, max_retries))
    g.add_node("format_check_loop", FormatCheckLoopNode(api_timeout, max_retries, get_guideline))
    g.add_node("emoji_check", EmojiCheckNode(api_timeout, max_retries))
    g.add_node("line_reduce", LineReduceNode())

    g.set_entry_point("detect_category")
    g.add_edge("detect_category", "format_check_loop")
    g.add_edge("format_check_loop", "emoji_check")
    g.add_edge("emoji_check", "line_reduce")
    g.add_edge("line_reduce", END)

    return g.compile()

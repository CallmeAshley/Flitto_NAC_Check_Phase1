# graph/file_graph.py — file-level LangGraph with JSONL error logging
from __future__ import annotations
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
import os, json, time
from threading import Lock

from graph.line_subgraph import build_line_subgraph
from utils.file_utils import get_guideline
from prompt_builder.build_prompt import (
    build_missing_check_prompt,
    build_addition_check_prompt,
)
from utils.gpt_client import ask_gpt5_async, ask_gpt4o_async
from utils.helper import b, llist, normalize_gpt_json

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

def _log_error_file(
    state_like: Dict[str, Any],
    *,
    stage: Optional[str],
    error_type: str,
    error_message: str,
) -> None:
    """
    Append a file-level error as one JSON line into error.jsonl
    """
    payload = {
        "type": "gpt_call_error",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "context": {
            "parent_folder": state_like.get("parent_folder", "unknown"),
            "filename": state_like.get("filename", "unknown"),
        },
        "stage": stage,
        "line_no": None,
        "category": None,
        "error": {"type": error_type, "message": error_message},
        "guideline": None,
    }
    _append_error_jsonl(payload, state_like.get("output_dir"))

async def _safe_ask(
    func,
    messages,
    *,
    model,
    timeout,
    max_retries,
    stage: str,
    state_for_log: Optional[Dict[str, Any]] = None,
):
    """
    Wrapper for GPT calls with JSONL error logging.
    On exception: logs one JSON record and returns ("error", {})
    """
    try:
        return await func(messages, model=model, timeout=timeout, max_retries=max_retries)
    except Exception as e:
        if state_for_log is not None:
            _log_error_file(
                state_for_log,
                stage=stage,
                error_type=type(e).__name__,
                error_message=str(e),
            )
        return "error", {}


class FileState(TypedDict, total=False):
    """State dict for one JSON file throughout the graph"""
    input_path: str
    parent_folder: str
    filename: str
    output_dir: str
    source: str
    target: str
    text: str
    trans: str
    src_lines: List[str]
    trn_lines: List[str]
    format_checked_lines: List[str]
    checked_sentences: List[dict]
    emoji_line_issues: List[dict]
    format_checked_text: str
    final_doc: str
    final_checked_joined: str
    res_missing: dict
    res_addition: dict
    semantic_issues: List[dict]
    API_TIMEOUT_SEC: int
    MAX_RETRIES: int
    CONCURRENCY_LINES: int
    failures: List[str]


class LoadFileNode:
    """Load JSON into FileState"""
    def __call__(self, s: FileState) -> FileState:
        st = s.copy()
        with open(st["input_path"], "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        st["source"] = data.get("source")
        st["target"] = data.get("target")
        st["text"]   = data.get("text", "") or ""
        st["trans"]  = data.get("trans", "") or ""
        st["src_lines"] = st["text"].splitlines()
        st["trn_lines"] = st["trans"].splitlines()
        N = max(len(st["src_lines"]), len(st["trn_lines"]))
        if len(st["src_lines"]) < N: st["src_lines"] += [""] * (N - len(st["src_lines"]))
        if len(st["trn_lines"]) < N: st["trn_lines"] += [""] * (N - len(st["trn_lines"]))
        st["format_checked_lines"] = [""] * N
        st["checked_sentences"] = []     
        st["emoji_line_issues"] = []      
        st["failures"] = []
        os.makedirs(os.path.join(st["output_dir"], st["parent_folder"]), exist_ok=True)
        return st


class MapLinesNode:
    """Map line-level subgraph over all lines, then merge results"""
    def __init__(self, api_timeout: int, max_retries: int, concurrency: int):
        self.subgraph = build_line_subgraph(api_timeout, max_retries, get_guideline)
        self.concurrency = concurrency

    async def __call__(self, s: FileState) -> FileState:
        st = s.copy()
        N = len(st["src_lines"])
        items = []
        for i in range(N):
            items.append({
                "i": i,
                "src_line": st["src_lines"][i].strip(),
                "trn_line": st["trn_lines"][i].strip(),
                "target": st["target"],
                "parent_folder": st["parent_folder"],
                "filename": st["filename"],
                "output_dir": st["output_dir"],
            })
        results = await self.subgraph.abatch(items, config={"executor": {"max_concurrency": self.concurrency}})
        for r in results:
            i = r["i"]
            st["format_checked_lines"][i] = r.get("revised_fmt", r.get("trn_line", ""))
            if r.get("checked_sentence_item"):
                st["checked_sentences"].append(r["checked_sentence_item"])
            if r.get("emoji_issue_item"):
                st["emoji_line_issues"].append(r["emoji_issue_item"])
        st["format_checked_text"] = "\n".join(st["format_checked_lines"])
        return st


class MissingCheckNode:
    async def __call__(self, s: FileState) -> FileState:
        st = s.copy()
        st["final_doc"] = st.get("format_checked_text", "\n".join(st.get("format_checked_lines", [])))

        if st["text"] and st["final_doc"]:
            sys2, usr2 = build_missing_check_prompt(st["text"], st["final_doc"])
            res, _ = await _safe_ask(
                ask_gpt5_async, [sys2, usr2],
                model="gpt-5", timeout=st["API_TIMEOUT_SEC"], max_retries=st["MAX_RETRIES"],
                stage="missing_check", state_for_log=st
            )
            js = normalize_gpt_json(res) if res != "error" else {}
            
            if len(js['suggestions']) > 1: 
                temp = ''
                for sug in js['suggestions']:
                    if isinstance(sug, str) and sug.strip():
                        temp += sug.strip()
                        temp += '\n'
                js['suggestions'] = [temp.strip()]
                
            st["res_missing"] = js or {"missing_content": False, "suggestions": []}

            suggestion = js.get("suggestions") if js['suggestions'] else None
            # sugs = js.get("suggestions") if js else None
            # if isinstance(sugs, list):
            #     for v in sugs:
            #         if isinstance(v, str) and v.strip():
            #             suggestion = v.strip()
            #             break

            if suggestion:
                expected = len((st.get("format_checked_text") or "").splitlines()) or 1
                suggested = suggestion[0].count("\n") + 1
                if suggested == expected:
                    st["final_doc"] = suggestion[0].rstrip("\n")
                else:
                    _log_error_file(
                        st,
                        stage="missing_check_line_count_mismatch",
                        error_type="LineCountMismatch",
                        error_message=f"expected={expected}, suggested={suggested}"
                    )
            else:
                st["res_missing"] = {"missing_content": False, "suggestions": []}
        else:
            st["res_missing"] = {"missing_content": False, "suggestions": []}
        return st


class AdditionCheckNode:
    """Document-level addition/faithfulness check"""
    async def __call__(self, s: FileState) -> FileState:
        st = s.copy()
        if st["text"] and st["final_doc"]:
            sys3, usr3 = build_addition_check_prompt(st["text"], st["final_doc"])
            res, _ = await _safe_ask(
                ask_gpt5_async, [sys3, usr3],
                model="gpt-5", timeout=st["API_TIMEOUT_SEC"], max_retries=st["MAX_RETRIES"],
                stage="addition_check", state_for_log=st
            )
            
            js = normalize_gpt_json(res) if res != "error" else {}
            
            
            if len(js['suggestions']) > 1: 
                temp = ''
                for sug in js['suggestions']:
                    if isinstance(sug, str) and sug.strip():
                        temp += sug.strip()
                        temp += '\n'
                js['suggestions'] = [temp.strip()]
    
            st["res_addition"] = js or {"faithfulness_issue": False, "suggestions": []}
            
            
            n_lines = len(st.get("text", "").splitlines()) or 1
            suggested = js.get("suggestions") if js['suggestions'] else None


            if suggested:
                if (suggested[0].count("\n") + 1 == n_lines):
                    st["final_checked_joined"] = suggested[0].rstrip("\n")
                    
                else:
                    _log_error_file(
                        st,
                        stage="addition_check_line_count_mismatch",
                        error_type="LineCountMismatch",
                        error_message=f"expected={n_lines}, suggested={suggested[0].count("\n") + 1}"
                    )
                    st["final_checked_joined"] = st["final_doc"].rstrip("\n")
            else:
                st["final_checked_joined"] = st["final_doc"].rstrip("\n")
                
        else:
            st["res_addition"] = {"faithfulness_issue": False, "suggestions": []}
            st["final_checked_joined"] = (st.get("final_doc") or st.get("format_checked_text") or "").rstrip("\n")
            
        return st


class FinalizeAndSaveNode:
    """Assemble semantic issues and save result JSON"""
    def __call__(self, s: FileState) -> FileState:
        st = s.copy()

        # === content_check ===
        emoji_issue_flag = len(st.get("emoji_line_issues", [])) > 0
        missing_issue = b(st.get("res_missing", {}).get("missing_content"), False)
        faith_issue = b(st.get("res_addition", {}).get("faithfulness_issue"), False)

        content_check = {
            "emoji_issue": emoji_issue_flag,
            "emoji_line_issues": st.get("emoji_line_issues", []), 
            "missing_content": missing_issue,
            "missing_spans": llist(st.get("res_missing", {}).get("missing_spans")),
            "revised_missing_spans": llist(st.get("res_missing", {}).get("revised_spans")),
            "faithfulness_issue": faith_issue,
            "added_spans": llist(st.get("res_addition", {}).get("added_spans"))
        }

        # === 결과 JSON 저장 ===
        out_folder = os.path.join(st["output_dir"], st["parent_folder"])
        os.makedirs(out_folder, exist_ok=True)
        output_path = os.path.join(out_folder, st["filename"])

        result_json = {
            "source": st["source"],
            "target": st["target"],
            "source_text": st["text"],
            "original_trans": st["trans"],
            "final_llm_suggestion": st["final_checked_joined"],  
            "format_check": st.get("checked_sentences", []),       
            "content_check": content_check                         
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        return st



def build_file_graph(
    *,
    API_TIMEOUT_SEC: int,
    MAX_RETRIES: int,
    CONCURRENCY_LINES: int,
):
    """Build and return compiled file-level LangGraph"""
    g = StateGraph(FileState)
    g.add_node("load_file", LoadFileNode())
    g.add_node("map_lines",  MapLinesNode(API_TIMEOUT_SEC, MAX_RETRIES, CONCURRENCY_LINES))
    g.add_node("missing_check", MissingCheckNode())
    g.add_node("addition_check", AdditionCheckNode())
    g.add_node("finalize_save", FinalizeAndSaveNode())
    g.set_entry_point("load_file")
    g.add_edge("load_file", "map_lines")
    g.add_edge("map_lines", "missing_check")
    g.add_edge("missing_check", "addition_check")
    g.add_edge("addition_check", "finalize_save")
    g.add_edge("finalize_save", END)
    return g.compile()

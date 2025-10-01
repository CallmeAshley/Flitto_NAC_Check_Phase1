# utils/file_utils.py — guideline cache + JSONL error logging when missing
import os, time, json
from threading import Lock

GUIDE_BASE_DIR = "/mnt/c/Users/Flitto/Documents/NAC/LLM검수/LCT_check_phase1/docs"
GUIDE_CACHE = {}  # key: (locale, category) -> str

_ERROR_LOG_LOCK = Lock()

def _error_log_path() -> str:
    base = os.getenv("OUTPUT_DIR") or os.getcwd()
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "error.jsonl")

def _append_error_jsonl(payload: dict) -> None:
    path = _error_log_path()
    line = json.dumps(payload, ensure_ascii=False)
    with _ERROR_LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def _log_guideline_missing(locale: str, category: str):
    payload = {
        "type": "guideline_missing",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "context": {"parent_folder": "N/A", "filename": None},
        "stage": None,
        "line_no": None,
        "category": None,
        "error": None,
        "guideline": {"locale": locale, "name": f"{category}.txt"},
    }
    _append_error_jsonl(payload)

def load_guideline(locale: str, category: str) -> str:
    """
    Load one guideline file. On missing file, logs a JSON record to error.jsonl
    and returns empty string.
    """
    file_path = os.path.join(GUIDE_BASE_DIR, locale, f"{category}.txt")
    if not os.path.exists(file_path):
        _log_guideline_missing(locale, category)
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_guideline(locale: str, category: str) -> str:
    """
    Cached guideline getter. Logs to error.jsonl when the file is missing.
    """
    key = (locale, category)
    if key in GUIDE_CACHE:
        return GUIDE_CACHE[key]

    file_path = os.path.join(GUIDE_BASE_DIR, locale, f"{category}.txt")
    if not os.path.exists(file_path):
        _log_guideline_missing(locale, category)
        GUIDE_CACHE[key] = ""
        return ""

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    GUIDE_CACHE[key] = content
    return content

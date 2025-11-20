# main.py — BE 스타일로 정리된 배치 엔트리포인트 (folders → files, per-file LangGraph)
import os
import re
import asyncio
from glob import glob
from typing import Optional, Dict

from graph.file_graph import build_file_graph

# ================== Settings ==================
INPUT_DIR = "/mnt/c/Users/Flitto/Documents/NAC/LLM검수/Advanced_async_batch/data/input2_json"
OUTPUT_DIR = "/mnt/c/Users/Flitto/Documents/NAC/LLM검수/LCT_check_phase1/data/시연"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SUBFOLDERS = [
    "NAC_3175_en-ko_HTL_7956_250313_170837_DONE",
    # "input_en-ko",
    # "NAC_2425_ko-en_HTL_405_250318_172641",
    # "NAC_2433_fr-en_HTL_389_250318_172605",
    # "NAC_2435_ar-en_HTL_412_250318_172556",
    # "NAC_3167_en-uk_HTL_7956_250313_174738_DONE",
]

MAX_FILES_PER_FOLDER: Optional[int] = None  # None이면 제한 없음

# Concurrency / retry (line-level은 file_graph 내부에서 사용)
# CONCURRENCY_LINES = 1
# API_TIMEOUT_SEC = 60
# MAX_RETRIES = 10

CONCURRENCY_LINES = 1
API_TIMEOUT_SEC = 3600           # API 레벨 타임아웃도 크게 (1시간)
MAX_RETRIES = 10

# ================== Utils ==================
def _natural_sort_key(path: str) -> int:
    """파일명 내 첫 숫자를 기준으로 정렬, 숫자가 없으면 매우 큰 값으로 뒤로."""
    base = os.path.basename(path)
    m = re.search(r"/d+", base)
    return int(m.group()) if m else 10**12


async def _process_single_file(
    input_json_path: str,
    *,
    output_dir: str,
    timeout: int,
    max_retries: int,
    concurrency: int,
) -> Dict[str, Optional[str]]:
    """
    Run the file-level graph for one JSON input.
    Side effects (file_graph 책임):
      - 결과 JSON: {output_dir}/{parent_folder}/{filename}
      - 에러 JSONL: {output_dir}/error.jsonl 에 append

    Returns:
        {
          "ok": bool,
          "output_path": str | None,
          "error_log": str
        }
    """
    parent_folder = os.path.basename(os.path.dirname(input_json_path)) or "unknown"
    filename = os.path.basename(input_json_path)

    file_graph = build_file_graph(
        API_TIMEOUT_SEC=timeout,
        MAX_RETRIES=max_retries,
        CONCURRENCY_LINES=concurrency,
    )

    state = {
        "input_path": input_json_path,
        "parent_folder": parent_folder,
        "filename": filename,
        "output_dir": output_dir,
        "API_TIMEOUT_SEC": timeout,
        "MAX_RETRIES": max_retries,
        "CONCURRENCY_LINES": concurrency,
    }

    # 실행 (체크포인트 비활성화)
    await file_graph.ainvoke(state, config={"execution": {"checkpoint": False}})

    # 산출물 경로 구성
    output_path = os.path.join(output_dir, parent_folder, filename)
    error_log = os.path.join(output_dir, "error.jsonl")
    ok = os.path.isfile(output_path)

    return {"ok": ok, "output_path": output_path if ok else None, "error_log": error_log}


async def _run_batch() -> None:
    for sub in TARGET_SUBFOLDERS:
        folder = os.path.join(INPUT_DIR, sub)
        if not os.path.isdir(folder):
            print(f"⚠️  Skipped (not found): {folder}")
            continue

        json_files = sorted(glob(os.path.join(folder, "*.json")), key=_natural_sort_key)
        if MAX_FILES_PER_FOLDER is not None:
            json_files = json_files[:MAX_FILES_PER_FOLDER]

        for fp in json_files:
            result = await _process_single_file(
                fp,
                output_dir=OUTPUT_DIR,
                timeout=API_TIMEOUT_SEC,
                max_retries=MAX_RETRIES,
                concurrency=CONCURRENCY_LINES,
            )
            
            if result["ok"]:
                print(f"✅ Processed: {sub}/{os.path.basename(fp)}")
            else:
                print(f"❌ Failed (no output): {sub}/{os.path.basename(fp)}  → see {result['error_log']}")


async def main() -> None:
    await _run_batch()


if __name__ == "__main__":
    asyncio.run(main()) 

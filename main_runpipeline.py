# main.py — single-file entrypoint with return contract and docstrings
import os
import asyncio
from graph.file_graph import build_file_graph

OUTPUT_DIR = "/mnt/c/Users/Flitto/Documents/NAC/LLM검수/LCT_check_phase1/data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def _process_single_file(
    input_json_path: str,
    output_dir: str,
    *,
    timeout: int,
    max_retries: int,
    concurrency: int,
) -> None:
    """
    Internal coroutine that runs the file-level graph for one JSON input.
    Side effects:
      - Writes result JSON under {output_dir}/{parent_folder}/{filename}
      - Appends error records (JSONL) to {output_dir}/error.jsonl
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
    await file_graph.ainvoke(state, config={"execution": {"checkpoint": False}})


def run_pipeline(
    input_json_path: str,
    output_dir: str = OUTPUT_DIR,
    *,
    timeout: int = 3600,
    max_retries: int = 10,
    concurrency: int = 1,
) -> dict:
    """
    Run the LCT check pipeline for exactly one input JSON.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_dir (str): Base output directory.
        timeout (int): GPT API timeout seconds.
        max_retries (int): Retry attempts for GPT calls.
        concurrency (int): Line-level concurrency.

    Returns:
        dict: {
            "ok": bool,                 # True if result JSON written
            "output_path": str | None,  # Path to result JSON
            "error_log": str            # Path to error.jsonl (JSON Lines)
        }
    """
    if not isinstance(input_json_path, str) or not os.path.isfile(input_json_path):
        return {
            "ok": False,
            "output_path": None,
            "error_log": os.path.join(output_dir, "error.jsonl"),
        }

    os.makedirs(output_dir, exist_ok=True)
    asyncio.run(
        _process_single_file(
            input_json_path,
            output_dir,
            timeout=timeout,
            max_retries=max_retries,
            concurrency=concurrency,
        )
    )

    parent_folder = os.path.basename(os.path.dirname(input_json_path)) or "unknown"
    filename = os.path.basename(input_json_path)
    output_path = os.path.join(output_dir, parent_folder, filename)
    error_log = os.path.join(output_dir, "error.jsonl")

    ok = os.path.isfile(output_path)
    return {"ok": ok, "output_path": output_path if ok else None, "error_log": error_log}

#!/usr/bin/env python3
"""
Formula Split Agent (LLM-only)

Processes a CSV where each model column contains a noisy textual list of formulas.
This agent splits formulas into separate per-model columns <model>_1, <model>_2, ...
based exclusively on an LLM extraction step. No local regex, parsing, or math is used
to infer formulas.

Behavior:
- Uses an LLM to separate formulas and return ONLY a JSON array of strings.
- Preserves each formula token content as returned by the LLM; the splitter does not
  rewrite or normalize text beyond wrapping values in double quotes.
- Determines column count per model by the max number of formulas across rows.
- Pads missing entries with "*not found"; propagates explicit not-found rows.
- Supports progressive saves to the same CSV with --save-every.
"""

import argparse
import logging
import re
import json
from typing import List, Optional

import pandas as pd

try:
    import ollama
except Exception:
    ollama = None


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ensure_quoted(formula: str) -> str:
    """Wrap the value in double quotes; do not alter internal content."""
    s = str(formula)
    if s.startswith('"') and s.endswith('"'):
        return s
    return f'"{s}"'


def llm_extract_formulas(text: str, model_name: str = 'llama3.1:8b', max_attempts: int = 2) -> List[str]:
    """Use a local LLM to extract formulas as a JSON list of strings.

    Strictly LLM-only: no local parsing or rewriting. The LLM must identify
    formulas and return them verbatim in a JSON array of strings. Returns []
    if no parseable result is produced or the LLM is unavailable.
    """
    if ollama is None:
        return []

    base_prompt = (
        "You are given a single cell containing a noisy textual list of formulas for a sequence.\n"
        "Task: Identify each individual formula and return ONLY a JSON array of strings.\n"
        "Important rules:\n"
        "- Preserve exact tokens and spacing for each formula; do not rewrite or normalize.\n"
        "- Lists may or may not be surrounded by brackets [ ... ].\n"
        "- Items may or may not be surrounded by quotes; treat unquoted items as formulas too.\n"
        "- The whole list may be inside quotes; still separate items.\n"
        "- If the cell clearly indicates no formulas (e.g., '*not found'), return [].\n"
        "- Return only the JSON array, no commentary.\n\n"
        "Cell:\n" + text
    )

    strict_prompt = (
        "Return ONLY a valid JSON array of strings representing the formulas from the cell.\n"
        "Do not include any text before or after the JSON.\n"
        "Preserve exact token content; do not rewrite.\n\n"
        "Cell:\n" + text
    )

    prompt = base_prompt
    attempts = 0
    while attempts < max_attempts:
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "top_p": 0.9}
            )
            content = response["message"]["content"].strip()
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        except Exception as e:
            logger.debug(f"LLM extraction attempt {attempts+1} failed: {e}")
        attempts += 1
        prompt = strict_prompt
    return []


# LLM-only approach: no local parsing graph; extraction is delegated to llm_extract_formulas.


def process_model_columns(df: pd.DataFrame, model_columns: List[str], csv_path: str,
                          default_ollama_model: str = 'llama3.1:8b', save_every: int = 0,
                          max_attempts: int = 1) -> pd.DataFrame:
    """Process the specified model columns, splitting formulas into per-model columns.

    - For each model column, compute max number of formulas across its rows.
    - Create columns <model>_1.._<model>_k.
    - Fill cells row-by-row; pad with "*not found".
    - Save progressively to the same CSV if save_every > 0.
    """
    for model in model_columns:
        logger.info(f"Processing model column: {model}")

        # Sequential per-row extraction and assignment with on-the-fly column creation
        values = df[model].tolist()
        # Ensure at least one target column exists to start with
        max_len = 1
        target_cols = [f"{model}_1"]
        if target_cols[0] not in df.columns:
            df[target_cols[0]] = ''

        logger.info(f"  Starting sequential row-by-row split for {model}")
        for idx, val in enumerate(values):
            src_text = '' if pd.isna(val) else str(val)

            # Log raw source
            try:
                raw_preview = src_text if src_text is not None else ''
                if len(raw_preview) > 240:
                    raw_preview = raw_preview[:240] + "..."
                logger.info(f"  [source] {model} row {idx + 1}: {repr(raw_preview)}")
            except Exception:
                logger.debug(f"  [source] {model} row {idx + 1}: logging failed")

            # Determine explicit not-found rows
            lower_src = src_text.strip().lower().strip('"').strip("'")
            is_not_found_row = lower_src in {"*not found", "not found"}

            # Extract formulas per row (LLM-only, sequential)
            formulas: List[str] = []
            if not is_not_found_row and src_text:
                try:
                    formulas = llm_extract_formulas(src_text, default_ollama_model, max_attempts=max_attempts)
                except Exception as e:
                    logger.debug(f"  [extract] {model} row {idx + 1}: error {e}")
                    formulas = []

            # Log extraction result
            try:
                logger.info(f"  [extract] {model} row {idx + 1}: found {len(formulas)} -> {formulas}")
            except Exception:
                logger.debug(f"  [extract] {model} row {idx + 1}: logging failed")

            # Expand columns on the fly if needed
            if len(formulas) > max_len:
                for i in range(max_len + 1, len(formulas) + 1):
                    new_col = f"{model}_{i}"
                    if new_col not in df.columns:
                        df[new_col] = ''
                    target_cols.append(new_col)
                max_len = len(formulas)

            # Assign values for this row
            assigned_pairs: List[tuple] = []
            for i in range(max_len):
                if is_not_found_row:
                    val_out = "*not found"
                else:
                    val_out = formulas[i] if i < len(formulas) else "*not found"
                quoted_val = ensure_quoted(val_out)
                df.at[idx, target_cols[i]] = quoted_val
                assigned_pairs.append((target_cols[i], quoted_val))

            # Log per-row assignments
            try:
                assignment_str = "; ".join([f"{col}={val}" for col, val in assigned_pairs])
                logger.info(f"  [assign] {model} row {idx + 1}: {assignment_str}")
            except Exception:
                logger.debug(f"  [assign] {model} row {idx + 1}: logging failed")

            # Checkpoint saves
            if save_every and (idx + 1) % save_every == 0:
                df.to_csv(csv_path, index=False)
                logger.info(f"  Saved checkpoint at row {idx + 1} for model {model}")

        # Save after completing the model
        df.to_csv(csv_path, index=False)
        logger.info(f"Completed model {model}: created columns {', '.join(target_cols)}")

    return df


def detect_model_columns(df: pd.DataFrame) -> List[str]:
    """Detect model columns by excluding known non-model columns and generated split columns."""
    non_model = {"sequence", "Complexity"}
    model_cols: List[str] = []
    for col in df.columns:
        if col in non_model:
            continue
        # Exclude already split columns (contain underscore and numeric suffix)
        if re.match(r"^.+_\d+$", col):
            continue
        model_cols.append(col)
    return model_cols


def main():
    parser = argparse.ArgumentParser(description="Formula Split Agent")
    parser.add_argument("--csv", default="/Users/alberto/Documents/projects/medLLM/multi-formula-time-series.csv",
                        help="Path to the CSV to process")
    parser.add_argument("--models", nargs="*", help="Specific model columns to process; default processes all models")
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N rows (0 disables)")
    parser.add_argument("--ollama-model", default="llama3.1:8b", help="Local Ollama model used for fallback extraction")
    parser.add_argument("--max-attempts", type=int, default=1, help="Max LLM attempts per cell")

    args = parser.parse_args()

    logger.info(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)

    # Normalize requested model names to match actual CSV headers
    def _norm_header(h: str) -> str:
        return str(h).strip().strip('"').strip("'")

    if args.models and len(args.models) > 0:
        header_map = {_norm_header(c): c for c in df.columns}
        resolved: List[str] = []
        for req in args.models:
            norm = _norm_header(req)
            if norm in header_map:
                resolved.append(header_map[norm])
            else:
                logger.warning(f"Requested model '{req}' not found among CSV headers. Available: {list(df.columns)}")
        model_cols = resolved if len(resolved) > 0 else detect_model_columns(df)
    else:
        model_cols = detect_model_columns(df)

    logger.info(f"Target model columns: {model_cols}")

    df = process_model_columns(
        df,
        model_cols,
        csv_path=args.csv,
        default_ollama_model=args.ollama_model,
        save_every=args.save_every,
        max_attempts=args.max_attempts,
    )

    logger.info("All requested model columns processed and saved.")


if __name__ == "__main__":
    main()
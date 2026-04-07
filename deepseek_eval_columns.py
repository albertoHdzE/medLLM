import argparse
import os
import re
import time
from typing import List, Optional

import ast
import math
import pandas as pd
import requests


def strip_outer_quotes(s: str) -> str:
    s = str(s).strip()
    if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def is_marker(s: str) -> bool:
    ss = str(s).strip()
    ssu = strip_outer_quotes(ss)
    return (ss.lower() in {"*not found", "not found"} or
            ssu.lower() in {"*not found", "not found"} or
            ssu.strip() in {"***"})


def _allowed_ast_node(node: ast.AST) -> bool:
    # Whitelist of safe nodes for arithmetic expressions in terms of 'n'
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.UnaryOp,
        ast.Constant,
        ast.Num,  # for older Python
        ast.Name,
        ast.Call,
        ast.Load,
        ast.Tuple,  # allow tuples in some expressions
        ast.List,   # allow list brackets if present
        ast.Subscript,
        ast.Index,
        ast.Attribute,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.IfExp,
    )
    return isinstance(node, allowed)


def _validate_ast(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not _allowed_ast_node(node):
            return False
    return True


def normalize_formula(formula: str) -> str:
    # Remove outer quotes and normalize caret to exponent
    f = strip_outer_quotes(str(formula).strip())
    # Replace '^' with '**' when used as exponent (common in user formulas)
    f = f.replace('^', '**')
    return f


def strip_lhs_assignment(expr: str) -> str:
    """Remove simple left-hand side assignments like 'a_n = 2*n' or 'a[n] = ...'.
    Returns the right-hand expression if an assignment is detected; otherwise returns expr.
    """
    # Match anything before '=' up to the first '=' and capture RHS
    m = re.match(r"^\s*[^=]+=\s*(.+)$", expr)
    if m:
        rhs = m.group(1).strip()
        return rhs
    return expr


def extract_assignments(expr: str) -> (str, dict):
    """Extract simple variable assignments from patterns like:
    - 'where m=3'
    - 'with m=3, k=2'
    - 'let m = 3'
    Returns cleaned expression and a dict of name->value (ints/floats).
    """
    cleaned = expr
    assignments = {}
    # Split on keywords and parse the right side for assignments
    # Support multiple assignments separated by commas
    m = re.search(r"\b(where|with|let)\b\s*(.*)$", cleaned, flags=re.IGNORECASE)
    if m:
        prefix = cleaned[:m.start()].strip()
        right = cleaned[m.end(1):].strip()
        # Split by commas for multiple assignments
        parts = [p.strip() for p in right.split(',') if p.strip()]
        for p in parts:
            mm = re.match(r"([a-zA-Z_]\w*)\s*=\s*([-+]?\d+(?:\.\d+)?)", p)
            if mm:
                name = mm.group(1)
                val = mm.group(2)
                if '.' in val:
                    try:
                        assignments[name] = float(val)
                    except Exception:
                        pass
                else:
                    try:
                        assignments[name] = int(val)
                    except Exception:
                        pass
        cleaned = prefix
    return cleaned, assignments


def _safe_eval_expression(expr: str, n_val: int) -> Optional[int]:
    """Safely evaluate an arithmetic expression of 'n' returning an int.
    Returns None if evaluation fails or yields non-finite value.
    """
    expr = normalize_formula(expr)
    expr = strip_lhs_assignment(expr)
    expr, assigns = extract_assignments(expr)
    if 'n' not in expr:
        return None
    try:
        tree = ast.parse(expr, mode='eval')
    except Exception:
        return None
    if not _validate_ast(tree):
        return None
    # Safe environment: allow only specific funcs
    safe_funcs = {
        'abs': abs,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
        'sqrt': math.sqrt,
        'pow': pow,
        'min': min,
        'max': max,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'exp': math.exp,
        'log': math.log,
    }
    safe_names = {'n': n_val, 'math': math}
    # Add extracted assignments
    for k, v in assigns.items():
        safe_names[k] = v
    try:
        val = eval(compile(tree, filename='<expr>', mode='eval'), {'__builtins__': {}}, {**safe_funcs, **safe_names})
    except Exception:
        return None
    # Convert to integer if appropriate
    if isinstance(val, (int,)):
        return int(val)
    if isinstance(val, float):
        if math.isfinite(val):
            # Accept near-integers
            r = round(val)
            if abs(val - r) < 1e-9:
                return int(r)
            # Otherwise cast to int if looks like integer progression
            return int(r)
        return None
    return None


def local_eval_line(expr: str, index_mode: str = '1-10') -> Optional[str]:
    """Try to evaluate a single formula locally for n range.
    Returns a quoted string of 10 integers or None on failure.
    """
    rng = list(range(1, 11)) if index_mode == '1-10' else list(range(0, 10))
    vals: List[int] = []
    for n_val in rng:
        v = _safe_eval_expression(expr, n_val)
        if v is None:
            return None
        vals.append(int(v))
    return '"' + ", ".join(str(x) for x in vals) + '"'


def deepseek_batch_eval(formulas: List[str], api_key: str, model: str = "deepseek-reasoner", index_mode: str = "1-10", timeout_sec: int = 180, max_retries: int = 3) -> List[str]:
    """Call DeepSeek chat completions API with a batch of formulas and return per-formula outputs.
    Each output is either a verbatim marker (e.g., '*not found', '***') or a quoted string of
    10 comma-and-space-separated integers representing evaluations for n indices 1..10.
    """
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }

    # Build instructions and list the formulas in order
    lines = []
    lines.append("You will receive a list of models or formulas, each representing a type of pseudocode or a mathematical/logical rule.")
    lines.append("Your task is to evaluate each formula to generate a list of integers.")
    lines.append("For each formula:")
    lines.append("- If it is a marker like *not found or ***, output that exact text verbatim.")
    if str(index_mode).strip() == "0-9":
        lines.append("- Otherwise, evaluate indices n=0..9 and produce exactly 10 integers.")
    else:
        lines.append("- Otherwise, evaluate indices n=1..10 and produce exactly 10 integers.")
    lines.append("- Output must be a single line enclosed in quotes, with values separated by a comma and a space. Example: \"0, 1, 2, 3, 4, 5, 6, 7, 8, 9\"")
    lines.append("Respond ONLY with one line per formula in the same order; no extra text.")
    lines.append("Formulas:")
    for i, f in enumerate(formulas, start=1):
        # Propagate markers in prompt listing; model should echo them verbatim
        lines.append(str(i) + ") " + str(f))

    prompt = "\n".join(lines)

    messages = [
        {"role": "system", "content": "You are a professional assistant. Respond ONLY with the output lines; no extra explanations."},
        {"role": "user", "content": prompt},
    ]

    data = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    attempt = 0
    while True:
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=timeout_sec)
        except requests.exceptions.RequestException:
            attempt += 1
            if attempt >= max_retries:
                return ["*not found" for _ in formulas]
            # brief backoff before retry
            time.sleep(2.0)
            continue
        if resp.status_code != 200:
            attempt += 1
            if attempt >= max_retries:
                return ["*not found" for _ in formulas]
            time.sleep(2.0)
            continue
        break

    result = {}
    try:
        result = resp.json()
    except Exception:
        return ["*not found" for _ in formulas]

    content = ""
    try:
        content = str(result["choices"][0]["message"]["content"]).strip()
    except Exception:
        return ["*not found" for _ in formulas]

    # Split by lines and map one line per formula
    out_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    # If the model concatenated outputs in a single paragraph, try splitting by quotes
    if len(out_lines) < len(formulas):
        parts = re.findall(r"\"[^\"]*\"|\*not found|\*\*\*", content)
        out_lines = [p.strip() for p in parts]

    # Normalize length
    if len(out_lines) < len(formulas):
        out_lines += ["*not found"] * (len(formulas) - len(out_lines))
    elif len(out_lines) > len(formulas):
        out_lines = out_lines[:len(formulas)]

    # Ensure each line is properly quoted or is a marker
    normalized: List[str] = []
    for ln in out_lines:
        lns = ln.strip()
        lnl = lns.lower()
        if lnl in {"*not found", "not found"} or lns.strip() in {"***"}:
            normalized.append(lns)
        else:
            lns_no_outer = strip_outer_quotes(lns)
            normalized.append("\"" + lns_no_outer + "\"")
    return normalized


def process_columns(csv_path: str, columns: List[str], batch_size: int = 90, deepseek_model: str = "deepseek-reasoner", timeout_sec: int = 180, max_retries: int = 3, save_every: int = 90, index_mode: str = '1-10') -> None:
    df = pd.read_csv(csv_path)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Warning: DEEPSEEK_API_KEY not set. Only local evaluation will run; remote formulas will be marked '*not found'.")

    for col in columns:
        if col not in df.columns:
            print(f"Column '{col}' not found; skipping.")
            continue
        eval_col = f"{col}_eval"
        if eval_col not in df.columns:
            df[eval_col] = ""

        # Prepare formulas list from this column
        formulas: List[str] = []
        for i in range(len(df)):
            raw = df.iloc[i][col]
            raw_str = str(raw).strip()
            raw_unquoted = strip_outer_quotes(raw_str)
            if pd.isna(raw) or raw_str == "":
                formulas.append("*not found")
            elif raw_str.lower() in {"*not found", "not found"} or raw_unquoted.lower() in {"*not found", "not found"} or raw_unquoted.strip() in {"***"}:
                formulas.append(raw_unquoted)
            else:
                formulas.append(raw_unquoted)

        # Process in batches (default 90 per your specification)
        total = len(formulas)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = formulas[start:end]
            print(f"Evaluating rows {start+1}-{end} of {total} for column '{col}'...")

            # Prepare outputs using local evaluator where possible
            outputs: List[Optional[str]] = [None] * len(batch)
            remote_formulas: List[str] = []
            remote_indices: List[int] = []

            for i, s in enumerate(batch):
                if is_marker(s):
                    outputs[i] = s
                    continue
                local = local_eval_line(s, index_mode=index_mode)
                if local is not None:
                    outputs[i] = local
                else:
                    remote_indices.append(i)
                    remote_formulas.append(s)

            if remote_formulas:
                # If there are any formulas needing remote evaluation
                if not api_key:
                    # Gracefully handle missing API key
                    rem_out = ["*not found" for _ in remote_formulas]
                else:
                    rem_out = deepseek_batch_eval(remote_formulas, api_key, model=deepseek_model, index_mode=index_mode, timeout_sec=timeout_sec, max_retries=max_retries)
                for idx, val in zip(remote_indices, rem_out):
                    outputs[idx] = val if val else "*not found"

            # Fill any remaining None with not found (shouldn't happen)
            outputs = [o if o is not None else "*not found" for o in outputs]

            for offset, out in enumerate(outputs):
                row_i = start + offset
                res = out if out else "*not found"
                df.at[row_i, eval_col] = res

            if (end % save_every) == 0:
                print(f"Saving checkpoint at row {end} for column '{col}'...")
                df.to_csv(csv_path, index=False)

        # Final save per column
        print(f"Finalizing column '{col}'...")
        df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate formulas via DeepSeek and write results to _eval columns.")
    parser.add_argument("--csv", default="multi-formula-time-series.csv", help="Path to CSV file containing formula columns")
    parser.add_argument("--columns", nargs="+", required=True, help="List of column names to process")
    parser.add_argument("--batch-size", type=int, default=90, help="Batch size per API call")
    parser.add_argument("--model", default="deepseek-reasoner", help="DeepSeek model name")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per request in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Max retries for API failures")
    parser.add_argument("--save-every", type=int, default=90, help="Save checkpoint every N rows")
    parser.add_argument("--index-mode", type=str, default="1-10", help="Index mode for evaluation: '1-10' or '0-9'.")
    args = parser.parse_args()

    process_columns(
        csv_path=args.csv,
        columns=args.columns,
        batch_size=args.batch_size,
        deepseek_model=args.model,
        timeout_sec=args.timeout,
        max_retries=args.retries,
        save_every=args.save_every,
        index_mode=args.index_mode,
    )


if __name__ == "__main__":
    main()
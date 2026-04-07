import argparse
import json
import math
import os
import re
from typing import List, Tuple, Optional as TypOptional, TypedDict

import pandas as pd
import requests

# LangGraph + Ollama fallback (guarded)
try:
    from langgraph.graph import StateGraph, END  # type: ignore
    from langgraph.errors import InvalidUpdateError  # type: ignore
except Exception:  # pragma: no cover
    StateGraph = None
    END = None
    class InvalidUpdateError(Exception):
        pass
try:
    from langchain_community.llms import Ollama  # type: ignore
except Exception:  # pragma: no cover
    Ollama = None

# Reuse the numeric expression evaluator
"""
Embedded NumericStringParser to avoid importing the package that triggers
multiprocessing Manager side-effects at import time.
Source adapted from Paul McGuire's pyparsing fourFn example.
"""
from pyparsing import (Literal, CaselessLiteral, Word, Combine, Group, Optional as PPOptional,
                       ZeroOrMore, Forward, nums, alphas, oneOf)


class NumericStringParser(object):
    def push_first(self, strg, loc, toks):
        self.exprStack.append(toks[0])

    def push_minus(self, strg, loc, toks):
        if toks and toks[0] == '-':
            self.exprStack.append('unary -')

    def __init__(self):
        point = Literal(".")
        e = CaselessLiteral("E")
        fnumber = Combine(Word("+-" + nums, nums) +
                          PPOptional(point + PPOptional(Word(nums))) +
                          PPOptional(e + Word("+-" + nums, nums)))
        ident = Word(alphas, alphas + nums + "_$")
        plus = Literal("+")
        minus = Literal("-")
        mult = Literal("*")
        div = Literal("/")
        lpar = Literal("(").suppress()
        rpar = Literal(")").suppress()
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")
        pi = CaselessLiteral("PI")
        expr = Forward()
        atom = ((PPOptional(oneOf("- +")) +
                 (ident + lpar + expr + rpar | pi | e | fnumber).setParseAction(self.push_first))
                | PPOptional(oneOf("- +")) + Group(lpar + expr + rpar)
                ).setParseAction(self.push_minus)
        factor = Forward()
        factor <<= atom + ZeroOrMore((expop + factor).setParseAction(self.push_first))
        term = factor + ZeroOrMore((multop + factor).setParseAction(self.push_first))
        expr <<= term + ZeroOrMore((addop + term).setParseAction(self.push_first))
        self.bnf = expr
        # map operator symbols to corresponding arithmetic operations
        self.opn = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '^': lambda a, b: a ** b,
        }
        self.fn = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'abs': abs,
            'round': round,
        }

    def evaluate_stack(self, s):
        op = s.pop()
        if op == 'unary -':
            return -self.evaluate_stack(s)
        if op in "+-*/^":
            op2 = self.evaluate_stack(s)
            op1 = self.evaluate_stack(s)
            return self.opn[op](op1, op2)
        elif op == "PI":
            return math.pi
        elif op == "E":
            return math.e
        elif op in self.fn:
            return self.fn[op](self.evaluate_stack(s))
        elif isinstance(op, str) and op and op[0].isalpha():
            # treat unknown identifiers as 0 to avoid NameError
            return 0
        else:
            return float(op)

    def eval(self, num_string, parseAll=True):
        self.exprStack = []
        self.bnf.parseString(num_string, parseAll)
        val = self.evaluate_stack(self.exprStack[:])
        return val


# Known base model columns provided by user
BASE_MODELS = [
    "chatgpt_4.5",
    "deepseek",
    "qwen",
    "gemini",
    "claude_3.5",
    "gpt_4o",
    "mistral",
    "meta",
    "o1_mini",
    "cursor_small",
    "o1_preview",
    "grok",
    "gemini-thinking",
    "grok_3",
    "claude_3.7",
    "gpt_4o_mini",
    "llama_4_scout",
    "qwen3",
    "chatgpt_5",
    "grok4",
    "deepseek_r1_0528",
    "opus_4",
    "mistral_large2405",
    "gemini_2.5_pro",
    "claude_sonnet_4",
]


def strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


# Removed normalization; we delegate interpretation to the LLM


def parse_sequence_string(seq_str: str) -> TypOptional[List[float]]:
    if seq_str is None:
        return None
    s = str(seq_str).strip()
    if not s:
        return None
    s = strip_outer_quotes(s)
    # Remove enclosing brackets if present
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    vals: List[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            # drop non-numeric entries safely
            return None
    return vals if vals else None


def safe_eval_expr(expr: str, n_value: int, nsp: NumericStringParser) -> TypOptional[float]:
    """Evaluate expression by substituting n with n_value using NumericStringParser."""
    try:
        sub = expr.replace("n", str(n_value))
        val = nsp.eval(sub)
        if isinstance(val, (int, float)):
            return float(val)
        return float(val)
    except Exception:
        return None


def expression_is_simple(expr: str) -> bool:
    """Allow only simple arithmetic over `n`: digits, spaces, (), + - * / ^, and 'n'.
    Reject functions or other identifiers to avoid non-integer outputs and long evaluations.
    """
    if not expr:
        return False
    # Only permit characters in the whitelist
    return re.fullmatch(r"[0-9n\s\+\-\*/\^().]+", expr) is not None


def all_close_to_integers(values: List[float], tol: float = 1e-9) -> bool:
    """Check all values are finite and within tolerance of an integer."""
    for v in values:
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return False
        if abs(float(v) - round(float(v))) > tol:
            return False
    return True


def format_values(values: List[float]) -> str:
    # Force integer-only representation; assume caller ensured values are near integers
    return ", ".join(str(int(round(v))) for v in values)


def generate_sequence_for_range(expr: str, rng: str, nsp: NumericStringParser) -> TypOptional[List[float]]:
    if not expr:
        return None
    # Reject complex expressions to avoid non-integer outputs or long evaluations
    if not expression_is_simple(expr):
        return None
    if rng == "0-9":
        ns = list(range(0, 10))
    elif rng == "1-10":
        ns = list(range(1, 11))
    else:
        return None
    vals: List[float] = []
    for n in ns:
        v = safe_eval_expr(expr, n, nsp)
        if v is None or not math.isfinite(v):
            return None
        vals.append(v)
    # Enforce integer-only sequences; if any value not near integer, consider not found
    if not all_close_to_integers(vals):
        return None
    return vals


def choose_best_range(expr: str, original: TypOptional[List[float]], nsp: NumericStringParser) -> str:
    """Pick 0-9 vs 1-10 based on matching original or feasibility."""
    seq_01 = generate_sequence_for_range(expr, "0-9", nsp)
    seq_12 = generate_sequence_for_range(expr, "1-10", nsp)

    # If both None, fallback heuristic
    if seq_01 is None and seq_12 is None:
        # Try n=0 evaluation feasibility
        v0 = safe_eval_expr(expr, 0, nsp)
        if v0 is None:
            return "1-10"
        return "0-9"

    if original and len(original) >= 10:
        def dist(a: List[float], b: List[float]) -> float:
            # L1 distance on first 10 values (or min length)
            m = min(len(a), len(b), 10)
            return sum(abs(a[i] - b[i]) for i in range(m))

        # Align lengths to 10 for fair comparison
        orig10 = original[:10]
        d01 = float("inf") if seq_01 is None else dist(seq_01, orig10)
        d12 = float("inf") if seq_12 is None else dist(seq_12, orig10)
        if d01 == d12:
            # tie-breaker: prefer feasible 1-10 to avoid zero pitfalls
            if seq_12 is not None:
                return "1-10"
            return "0-9"
        return "0-9" if d01 < d12 else "1-10"

    # Without original, prefer feasible range
    if seq_12 is not None:
        return "1-10"
    if seq_01 is not None:
        return "0-9"
    # Neither feasible; default to 1-10
    return "1-10"


class EvalState(TypedDict):
    text: str
    original: TypOptional[List[float]]
    result_str: TypOptional[str]


def build_graph(ollama_model: str):
    if Ollama is None or StateGraph is None:
        raise RuntimeError("Ollama or LangGraph not available in this environment")
    llm = Ollama(model=ollama_model)

    def llm_eval_node(state: EvalState) -> EvalState:
        expr = strip_outer_quotes(state.get("text", "")).strip()
        # Propagate common marker strings verbatim
        lower = expr.lower()
        if lower in {"*not found", "not found"} or expr.strip() in {"***"}:
            return {"result_str": expr}
        # Single-formula prompt: evaluate indices n=0..9 and return quoted, comma+space separated integers
        prompt = (
            "You will receive one formula representing a type of pseudocode or a mathematical/logical rule.\n"
            "Your task is to evaluate this formula to generate a list of integers.\n"
            "Evaluate the formula strictly for indices n=0..9 (ten evaluations).\n"
            "Output exactly one row containing the ten integer results, with the digits enclosed in quotes,\n"
            "and separated by a comma and a blank (space).\n"
            "For example: \"0, 1, 2, 3, 4, 5, 6, 7, 8, 9\"\n"
            "If the input contains a marker text such as *not found or *** or similar indicators, respond with that exact text.\n"
            f"Formula: {expr}"
        )

        try:
            out = llm.invoke(prompt)
            out_str = str(out).strip()
            # Minimal cleaning: strip outer quotes or code fences
            cleaned = strip_outer_quotes(out_str)
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned.strip("`").strip()
            while cleaned and cleaned[0] in ['`', '"', "'"]:
                cleaned = cleaned[1:]
            while cleaned and cleaned[-1] in ['`', '"', "'"]:
                cleaned = cleaned[:-1]
            # Ensure the result is wrapped in double quotes
            return {"result_str": f'"{cleaned}"' if cleaned else cleaned}
        except Exception:
            # Fall back to the original expression if it's a marker, else *not found
            if lower in {"*not found", "not found"} or expr.strip() in {"***"}:
                return {"result_str": expr}
            return {"result_str": "*not found"}

    graph = StateGraph(EvalState)
    graph.add_node("llm_eval", llm_eval_node)
    graph.set_entry_point("llm_eval")
    graph.add_edge("llm_eval", END)
    return graph.compile()


def deepseek_batch_eval(formulas: List[str], api_key: str, model: str, index_mode: str = "0-9", timeout_sec: int = 60, max_retries: int = 3) -> List[str]:
    """Call DeepSeek chat completions API with a batch of formulas and return per-formula outputs.
    Each output is either a verbatim marker (e.g., '*not found', '***') or a quoted string of
    10 comma-and-space-separated integers representing evaluations for n indices.
    """
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }

    # Build instructions and list the formulas in order
    lines = []
    lines.append("Evaluate these formulas. For each formula:")
    lines.append("- If it is a marker like *not found or ***, output that exact text verbatim.")
    if index_mode == "0-9":
        lines.append("- Otherwise, evaluate indices n=0..9 and produce exactly 10 integers.")
    else:
        lines.append("- Otherwise, evaluate indices n=1..10 and produce exactly 10 integers.")
    lines.append("- Output must be a single line enclosed in quotes, with values separated by a comma and a space. Example: \"0, 1, 2, 3, 4, 5, 6, 7, 8, 9\"")
    lines.append("- Respond ONLY with one line per formula in the same order; no extra text.")
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

    # Execute request with retry and timeout handling
    attempt = 0
    while True:
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=timeout_sec)
        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt >= max_retries:
                # On repeated failure, mark all as not found
                return ["*not found" for _ in formulas]
            continue
        if resp.status_code != 200:
            attempt += 1
            if attempt >= max_retries:
                return ["*not found" for _ in formulas]
            continue
        break
    if resp.status_code != 200:
        # On API failure, mark all as not found
        return ["*not found" for _ in formulas]

    result = resp.json()
    content = ""
    try:
        content = str(result["choices"][0]["message"]["content"]).strip()
    except Exception:
        return ["*not found" for _ in formulas]

    # Split by lines and map one line per formula
    out_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    # If the model concatenated outputs in a single paragraph, try splitting by quotes
    if len(out_lines) < len(formulas):
        # Attempt to extract quoted segments
        parts = re.findall(r"\"[^\"]*\"|\*not found|\*\*\*", content)
        out_lines = [p.strip() for p in parts]

    # Normalize length
    if len(out_lines) < len(formulas):
        # Pad missing outputs
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
            # Wrap in quotes if not already
            lns_no_outer = strip_outer_quotes(lns)
            normalized.append("\"" + lns_no_outer + "\"")
    return normalized


def detect_split_columns(df: pd.DataFrame,
                         models_filter: TypOptional[List[str]] = None,
                         columns_filter: TypOptional[List[str]] = None) -> List[Tuple[str, str]]:
    """Return list of (model, column_name) for split columns like '<model>_k'.
    If columns_filter is provided, only include columns listed there.
    """
    base_set = set(models_filter) if models_filter else set(BASE_MODELS)
    columns_set = set(columns_filter) if columns_filter else None
    results: List[Tuple[str, str]] = []
    for col in df.columns:
        if columns_set is not None and col not in columns_set:
            continue
        m = re.match(r"^(.+?)_(\d+)$", col)
        if not m:
            continue
        base = m.group(1)
        if base in base_set:
            results.append((base, col))
    return results


def process_csv(csv_path: str,
                models: TypOptional[List[str]],
                save_every: int,
                provider: str = "deepseek",
                deepseek_api_key: TypOptional[str] = None,
                deepseek_model: str = "deepseek-reasoner",
                batch_size: int = 90,
                index_mode: str = "0-9",
                ollama_model: str = "llama3.1:8b",
                timeout_sec: int = 60,
                max_retries: int = 3,
                columns: TypOptional[List[str]] = None) -> None:
    df = pd.read_csv(csv_path)
    use_deepseek = provider.lower() == "deepseek"
    app = None
    if not use_deepseek:
        # Build Ollama graph only if requested
        app = build_graph(ollama_model)

    # Attempt to parse original sequence column if present
    has_sequence = "sequence" in df.columns
    split_cols = detect_split_columns(df, models_filter=models, columns_filter=columns)
    print(f"Found {len(split_cols)} split columns to evaluate.")

    for idx, (base, col) in enumerate(split_cols):
        eval_col = f"{col}_eval"
        if eval_col not in df.columns:
            df[eval_col] = ""
        print(f"[{idx+1}/{len(split_cols)}] Processing model '{base}' column '{col}' -> '{eval_col}'.")
        if use_deepseek:
            # Batch DeepSeek processing
            # Prepare list of formulas in column order
            formulas: List[str] = []
            markers_idx: List[int] = []
            for row_i in range(len(df)):
                raw = df.iloc[row_i][col]
                raw_str = str(raw).strip()
                raw_unquoted = strip_outer_quotes(raw_str)
                if pd.isna(raw) or raw_str == "":
                    formulas.append("*not found")
                    markers_idx.append(row_i)
                elif raw_str.lower() in {"*not found", "not found"} or raw_unquoted.lower() in {"*not found", "not found"} or raw_unquoted.strip() in {"***"}:
                    formulas.append(raw_unquoted)
                    markers_idx.append(row_i)
                else:
                    formulas.append(raw_unquoted)

            # Process in batches
            total_rows = len(df)
            for start in range(0, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                batch = formulas[start:end]
                print("  Rows " + str(start + 1) + "/" + str(total_rows) + " to " + str(end) + "/" + str(total_rows) + "...")
                if not deepseek_api_key:
                    # Try environment fallback
                    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
                if not deepseek_api_key:
                    print("    Missing DeepSeek API key; marking batch as *not found.")
                    outputs = ["*not found" for _ in batch]
                else:
                    # Fast path: if every item is a marker, skip API call
                    all_markers = True
                    for s in batch:
                        ss = str(s).strip()
                        ssu = strip_outer_quotes(ss)
                        if not (ss.lower() in {"*not found", "not found"} or ssu.lower() in {"*not found", "not found"} or ssu.strip() in {"***"}):
                            all_markers = False
                            break
                    if all_markers:
                        outputs = batch
                    else:
                        outputs = deepseek_batch_eval(batch, deepseek_api_key, deepseek_model, index_mode, timeout_sec=timeout_sec, max_retries=max_retries)

                # Write outputs
                for offset, out in enumerate(outputs):
                    row_i = start + offset
                    res = out if out else "*not found"
                    print("    Formula: " + str(formulas[row_i]))
                    print("    Eval result: " + str(res))
                    df.at[row_i, eval_col] = res

                # Progressive save according to save_every only
                if (end % save_every) == 0:
                    print("  Saving checkpoint at rows " + str(end) + "...")
                    df.to_csv(csv_path, index=False)
        else:
            # Per-row Ollama fallback
            for row_i in range(len(df)):
                raw = df.iloc[row_i][col]
                if row_i % save_every == 0:
                    print(f"  Row {row_i+1}/{len(df)}...")
                raw_str = str(raw).strip()
                raw_unquoted = strip_outer_quotes(raw_str)
                # Propagate marker strings verbatim when detected
                if pd.isna(raw) or raw_str == "":
                    df.at[row_i, eval_col] = "*not found"
                    continue
                if raw_str.lower() in {"*not found", "not found"} or raw_unquoted.lower() in {"*not found", "not found"} or raw_unquoted.strip() in {"***"}:
                    df.at[row_i, eval_col] = raw_unquoted
                    print(f"    Marker detected; eval result: {raw_unquoted}")
                    continue

                original = None
                if has_sequence:
                    original = parse_sequence_string(df.iloc[row_i]["sequence"])  # type: ignore

                state: EvalState = {
                    "text": str(raw),
                    "original": original,
                    "result_str": None,
                }
                # Show the raw formula content being processed
                print(f"    Formula: {state['text']}")
                try:
                    final = app.invoke(state)  # type: ignore
                except InvalidUpdateError:
                    # If graph update fails, mark as not found
                    print("    Graph update failed; marking as *not found.")
                    df.at[row_i, eval_col] = "*not found"
                    continue

                res = final.get("result_str")
                if res:
                    print(f"    Eval result: {res}")
                else:
                    print(f"    Eval result: *not found")
                df.at[row_i, eval_col] = res if res else "*not found"

                # Progressive save
                if (row_i + 1) % save_every == 0:
                    print(f"  Saving checkpoint at row {row_i+1}...")
                    df.to_csv(csv_path, index=False)

        # Save after finishing one column
        print(f"Finished column '{col}'. Saving...")
        df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate split formula columns and write *_eval outputs.")
    parser.add_argument("--csv", type=str, default="/Users/alberto/Documents/projects/medLLM/multi-formula-time-series.csv",
                        help="Path to the CSV file to process.")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated base models to process (default: all known).")
    parser.add_argument("--columns", type=str, default="",
                        help="Comma-separated split columns to process (e.g., 'deepseek_1,gemini_2').")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save progress every N rows.")
    parser.add_argument("--provider", type=str, default="deepseek",
                        help="Provider to use: 'deepseek' or 'ollama'.")
    parser.add_argument("--deepseek-model", type=str, default="deepseek-reasoner",
                        help="DeepSeek model: 'deepseek-reasoner' or 'deepseek-chat'.")
    parser.add_argument("--deepseek-api-key", type=str, default=os.getenv("DEEPSEEK_API_KEY", ""),
                        help="DeepSeek API key; if omitted, reads DEEPSEEK_API_KEY env.")
    parser.add_argument("--batch-size", type=int, default=90,
                        help="Batch size for DeepSeek evaluations.")
    parser.add_argument("--index-mode", type=str, default="0-9",
                        help="Index mode for evaluation: '0-9' or '1-10'.")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b",
                        help="Local Ollama model name for fallback.")
    parser.add_argument("--timeout-sec", type=int, default=60,
                        help="Timeout in seconds for DeepSeek API requests.")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum retries for DeepSeek API requests.")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    columns = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None

    process_csv(
        csv_path=args.csv,
        models=models,
        save_every=args.save_every,
        provider=args.provider,
        deepseek_api_key=args.deepseek_api_key if args.deepseek_api_key else None,
        deepseek_model=args.deepseek_model,
        batch_size=args.batch_size,
        index_mode=args.index_mode,
        ollama_model=args.ollama_model,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        columns=columns,
    )


if __name__ == "__main__":
    main()
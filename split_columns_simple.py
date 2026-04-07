import argparse
import ast
import pandas as pd


def parse_items(cell: str):
    s = str(cell).strip()
    if s.lower() in {"*not found", "not found"}:
        return []
    # Try to parse as Python list first
    try:
        if s.startswith("[") and s.endswith("]"):
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return [str(x).strip() for x in val]
    except Exception:
        pass
    # Fallback: split by commas inside outer quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",")]
    return parts if len(parts) > 1 else [s]


def ensure_quoted(text: str) -> str:
    t = str(text)
    if t.startswith('"') and t.endswith('"'):
        return t
    return f'"{t}"'


def split_columns(csv_path: str, columns: list, save_every: int = 0):
    df = pd.read_csv(csv_path)
    for col in columns:
        if col not in df.columns:
            print(f"Column '{col}' not found; skipping.")
            continue
        print(f"Splitting column '{col}'...")
        # Determine max item count
        max_len = 1
        parsed_rows = []
        for i, val in enumerate(df[col].tolist()):
            items = parse_items(val)
            parsed_rows.append(items)
            if len(items) > max_len:
                max_len = len(items)

        # Create target columns
        target_cols = [f"{col}_{i}" for i in range(1, max_len + 1)]
        for t in target_cols:
            if t not in df.columns:
                df[t] = ''

        # Fill rows
        for i, items in enumerate(parsed_rows):
            for j in range(max_len):
                val_out = items[j] if j < len(items) else "*not found"
                df.at[i, target_cols[j]] = ensure_quoted(val_out)
            if save_every and (i + 1) % save_every == 0:
                df.to_csv(csv_path, index=False)
                print(f"Saved checkpoint at row {i+1} for '{col}'.")

        # Final save per column
        df.to_csv(csv_path, index=False)
        print(f"Finished splitting '{col}'; created: {', '.join(target_cols)}")


def main():
    parser = argparse.ArgumentParser(description="Simple deterministic splitter for list-like columns")
    parser.add_argument("--csv", required=True, help="Path to CSV")
    parser.add_argument("--columns", nargs="+", required=True, help="Columns to split")
    parser.add_argument("--save-every", type=int, default=0, help="Checkpoint save interval")
    args = parser.parse_args()

    split_columns(args.csv, args.columns, args.save_every)


if __name__ == "__main__":
    main()
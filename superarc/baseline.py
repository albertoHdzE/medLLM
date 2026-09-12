"""Freeze and verify the published state of the repository.

The baseline is the permanent reference that every later run is checked against.
It records three things:

  * ``artifacts``  — sha256 of every published figure file
  * ``inputs``     — sha256 of every input CSV
  * ``columns``    — sha256 of each individual column of the five datasets, so a
                     drift report can name the exact model that moved rather than
                     just saying "the file changed"

Column hashing is what makes the append-only rule enforceable: adding a model
adds columns but must leave every pre-existing column hash untouched.

Usage::

    python -m superarc.baseline freeze    # write data/baseline/
    python -m superarc.baseline verify    # compare working tree against it
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import REPO_ROOT

BASELINE_DIR = REPO_ROOT / "data" / "baseline"

# Published figure artifacts. Two directories, because the paper draws from both:
# new_plots/ for main-paper Figs 3-10, plots/highResolution/ for Fig 1 and the SI.
ARTIFACT_GLOBS = [
    "new_plots/*.pdf",
    "new_plots/*.png",
    "plots/highResolution/*.pdf",
    "plots/highResolution/*.svg",
    "plots/evolution/*.png",
    "plots/multi_formula/*.csv",
    "plots/multi_script/*.csv",
]

# The five datasets, with the encoding each must be read under. Getting these
# wrong silently mangles non-ASCII cells, so they are declared once here.
DATASETS = {
    "A_multi_formula": ("multi-formula-time-series.csv", "latin1"),
    "B_multi_script": ("multi-python-script-time-series.csv", "utf-8"),
    "C_series_ext": ("seriesWithLLMs_ext_Dic2025.csv", "latin-1"),
    "D_formulas_found": ("formulas_found_by_llm.csv", "utf-8"),
    "E_model_summary": ("comparison_models_summary.csv", "utf-8"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_series(series: pd.Series) -> str:
    """Hash a column by its values, independent of dtype inference.

    Values are rendered with ``repr`` of the Python object so that 1 and 1.0
    hash differently -- a dtype change is a real change and must be reported.
    """
    h = hashlib.sha256()
    for value in series.tolist():
        h.update(repr(value).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def collect_artifacts() -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in ARTIFACT_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if path.is_file():
                out[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    return out


def collect_inputs() -> dict[str, str]:
    return {
        str(p.relative_to(REPO_ROOT)): sha256_file(p)
        for p in sorted(REPO_ROOT.glob("*.csv"))
        if p.is_file()
    }


def collect_columns() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, (filename, encoding) in DATASETS.items():
        path = REPO_ROOT / filename
        if not path.exists():
            out[key] = {"file": filename, "present": False}
            continue
        df = pd.read_csv(path, encoding=encoding, low_memory=False)
        out[key] = {
            "file": filename,
            "encoding": encoding,
            "present": True,
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": {str(c): sha256_series(df[c]) for c in df.columns},
        }
    return out


def environment() -> dict:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return None

    packages = {}
    for name in ("pandas", "numpy", "matplotlib", "seaborn", "altair", "pybdm", "joblib"):
        try:
            packages[name] = __import__("importlib.metadata", fromlist=["version"]).version(name)
        except Exception:
            packages[name] = None

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "packages": packages,
    }


def freeze() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": environment(),
        "artifacts": collect_artifacts(),
        "inputs": collect_inputs(),
        "columns": collect_columns(),
    }
    for name in ("manifest", "artifacts", "inputs", "columns"):
        (BASELINE_DIR / f"{name}.json").write_text(
            json.dumps(payload[name], indent=2, sort_keys=True) + "\n"
        )

    n_cols = sum(
        len(v.get("columns", {})) for v in payload["columns"].values() if v.get("present")
    )
    print(f"baseline written to {BASELINE_DIR.relative_to(REPO_ROOT)}")
    print(f"  artifacts : {len(payload['artifacts'])}")
    print(f"  inputs    : {len(payload['inputs'])}")
    print(f"  columns   : {n_cols} across {len(payload['columns'])} datasets")
    print(f"  git       : {payload['manifest']['git_sha']} ({payload['manifest']['git_branch']})")


def verify() -> int:
    """Compare the working tree against the frozen baseline.

    Returns a process exit code: 0 when every pre-existing artifact, input and
    column is unchanged. Added columns are reported but are not failures --
    appending a model is the supported way to extend the datasets.
    """
    if not (BASELINE_DIR / "columns.json").exists():
        print("no baseline found; run `python -m superarc.baseline freeze` first")
        return 2

    old_artifacts = json.loads((BASELINE_DIR / "artifacts.json").read_text())
    old_inputs = json.loads((BASELINE_DIR / "inputs.json").read_text())
    old_columns = json.loads((BASELINE_DIR / "columns.json").read_text())

    failures: list[str] = []
    notes: list[str] = []

    for label, old, new in (
        ("artifact", old_artifacts, collect_artifacts()),
        ("input", old_inputs, collect_inputs()),
    ):
        for path, digest in old.items():
            if path not in new:
                failures.append(f"{label} MISSING: {path}")
            elif new[path] != digest:
                failures.append(f"{label} CHANGED: {path}")
        for path in new.keys() - old.keys():
            notes.append(f"{label} added: {path}")

    new_columns = collect_columns()
    for key, old_ds in old_columns.items():
        if not old_ds.get("present"):
            continue
        new_ds = new_columns.get(key, {})
        if not new_ds.get("present"):
            failures.append(f"dataset MISSING: {old_ds['file']}")
            continue
        if new_ds["n_rows"] != old_ds["n_rows"]:
            failures.append(
                f"dataset ROWS CHANGED: {old_ds['file']} "
                f"{old_ds['n_rows']} -> {new_ds['n_rows']}"
            )
        for col, digest in old_ds["columns"].items():
            if col not in new_ds["columns"]:
                failures.append(f"column MISSING: {old_ds['file']}::{col}")
            elif new_ds["columns"][col] != digest:
                failures.append(f"column CHANGED: {old_ds['file']}::{col}")
        added = new_ds["columns"].keys() - old_ds["columns"].keys()
        if added:
            notes.append(f"{old_ds['file']}: {len(added)} column(s) added")

    for note in notes:
        print(f"  note: {note}")
    if failures:
        print(f"\nDRIFT DETECTED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nno drift: every baselined artifact, input and column is unchanged")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    command = argv[0] if argv else "verify"
    if command == "freeze":
        freeze()
        return 0
    if command == "verify":
        return verify()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

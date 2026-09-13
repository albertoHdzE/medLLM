"""The single source of truth for model identity.

Why this exists
---------------
The same model appears under a different column name in every dataset, and each
producer script carried its own hand-written 28-entry rename dictionary -- six of
them in total. Keeping six copies in step by hand failed twice, in ways that
reached print:

* ``gemini``. Two maps inside 31-1_multiScript_experiment.py disagreed. Figure 5
  labelled the ``gemini-2.5-pro`` column "Gemini-2.5-Pro" and silently dropped
  ``gemini``; Figure 6 labelled the ``gemini`` column "Gemini-2.5-Pro" and
  silently dropped ``gemini-2.5-pro``. The two columns hold very different data
  (accuracy 9.70/3.03/0.00 versus 68.33/56.67/0.00), so the same label denotes
  different models in the two published figures. Resolved by the authors in
  favour of Figure 5: ``gemini-2.5-pro`` is Gemini-2.5-Pro.

* ``deepseek_r1_0528``. Published as "DeepSeek-R1-0525" in Figures 3 and 4 and as
  "DeepSeek-R1-0528" in Table 1 and Figures 7-10. One model, two labels. The
  column names are themselves swapped between datasets: the formula and series
  datasets call it ``deepseek_r1_0528`` while the script dataset calls it
  ``deepseek_r1_0525``. Table 1 is taken as authoritative here.

Silent dropping is what let both defects survive: every producer filtered with
``if get_model_display_name(model) != model``, so any column missing from the map
disappeared from the figure without a word. :func:`validate_columns` replaces
that with a loud failure, and retirement must now be stated explicitly.

Adding a model is one entry in ``MODELS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Dataset keys. The file each refers to is declared in superarc.baseline.DATASETS.
A_FORMULA = "A_formula"   # multi-formula-time-series.csv          -> Figs 3, 4
B_SCRIPT = "B_script"     # multi-python-script-time-series.csv    -> Figs 5, 6
C_SERIES = "C_series"     # seriesWithLLMs_ext_Dic2025.csv         -> Table 1, Figs 7-10


@dataclass(frozen=True)
class Model:
    """One evaluated model, and the column it occupies in each dataset.

    ``name`` is the label as printed in the paper. ``order`` is the position
    within the family, oldest first, and drives the family-evolution figures.
    A missing alias means the model was not run on that experiment.
    """

    name: str
    family: str
    order: int
    aliases: dict[str, str] = field(default_factory=dict)

    def column(self, dataset: str) -> str | None:
        return self.aliases.get(dataset)


MODELS: tuple[Model, ...] = (
    # --- OpenAI -----------------------------------------------------------
    Model("ChatGPT-4o", "OpenAI", 0,
          {A_FORMULA: "gpt_4o", B_SCRIPT: "chatgpt-4o", C_SERIES: "gpt_4o"}),
    Model("ChatGPT-4o-Mini", "OpenAI", 1,
          {A_FORMULA: "gpt_4o_mini", B_SCRIPT: "gpt-4o-mini", C_SERIES: "gpt_4o_mini"}),
    Model("ChatGPT-4.5", "OpenAI", 2,
          {A_FORMULA: "chatgpt_4.5", B_SCRIPT: "chatgpt-4.5", C_SERIES: "chatgpt_4.5"}),
    Model("o1-Preview", "OpenAI", 3,
          {A_FORMULA: "o1_preview", B_SCRIPT: "chatgpt-o1", C_SERIES: "o1_preview"}),
    Model("o1-Mini", "OpenAI", 4,
          {A_FORMULA: "o1_mini", B_SCRIPT: "o1_mini", C_SERIES: "o1_mini"}),
    Model("ChatGPT-5", "OpenAI", 5,
          {A_FORMULA: "chatgpt_5", B_SCRIPT: "chatgpt_5", C_SERIES: "chatgpt_5"}),
    Model("ChatGPT-5.2", "OpenAI", 6,
          {A_FORMULA: "gpt-5.2", B_SCRIPT: "gpt-5.2", C_SERIES: "gpt-5.2"}),

    # --- Google -----------------------------------------------------------
    # The B_SCRIPT alias for Gemini is `gemini-thinking`, not `gemini`;
    # see RETIRED_COLUMNS below.
    Model("Gemini", "Gemini", 0,
          {A_FORMULA: "gemini", B_SCRIPT: "gemini-thinking", C_SERIES: "gemini"}),
    Model("Gemini-2.5-Pro", "Gemini", 1,
          {A_FORMULA: "gemini_2.5_pro", B_SCRIPT: "gemini-2.5-pro", C_SERIES: "gemini_2.5_pro"}),
    Model("Gemini-3-Pro", "Gemini", 2,
          {A_FORMULA: "gemini-3-pro", B_SCRIPT: "gemini-3-pro", C_SERIES: "gemini-3-pro"}),

    # --- Anthropic --------------------------------------------------------
    Model("Claude-3.5", "Claude", 0,
          {A_FORMULA: "claude_3.5", B_SCRIPT: "claude-3-5-sonnet", C_SERIES: "claude_3.5"}),
    Model("Claude-3.7", "Claude", 1,
          {A_FORMULA: "claude_3.7", B_SCRIPT: "claude-3.7", C_SERIES: "claude_3.7"}),
    Model("Claude-Sonnet-4", "Claude", 2,
          {A_FORMULA: "claude_sonnet_4", B_SCRIPT: "claude_sonnet_4", C_SERIES: "claude_sonnet_4"}),
    Model("Claude-Opus-4", "Claude", 3,
          {A_FORMULA: "opus_4", B_SCRIPT: "opus_4", C_SERIES: "opus_4"}),
    Model("Claude-4.5", "Claude", 4,
          {A_FORMULA: "claude-4.5", B_SCRIPT: "claude-4.5", C_SERIES: "claude-4.5"}),

    # --- xAI --------------------------------------------------------------
    # The script dataset's column names lag the model versions by one: its
    # `grok` column is Grok-3 and its `grok-3` column is Grok-4.
    Model("Grok-3", "Grok", 0,
          {A_FORMULA: "grok_3", B_SCRIPT: "grok", C_SERIES: "grok_3"}),
    Model("Grok-4", "Grok", 1,
          {A_FORMULA: "grok4", B_SCRIPT: "grok-3", C_SERIES: "grok4"}),
    Model("Grok-4.1", "Grok", 2,
          {A_FORMULA: "grok_4.1", B_SCRIPT: "grok_4.1", C_SERIES: "grok_4.1"}),

    # --- Mistral ----------------------------------------------------------
    Model("Mistral", "Mistral", 0,
          {A_FORMULA: "mistral", B_SCRIPT: "mistral", C_SERIES: "mistral"}),
    Model("Mistral-Large-2405", "Mistral", 1,
          {A_FORMULA: "mistral_large2405", B_SCRIPT: "mistral_large2405", C_SERIES: "mistral_large2405"}),
    Model("Mistral-Large-3", "Mistral", 2,
          {A_FORMULA: "mistral-large-3", B_SCRIPT: "mistral-large-3", C_SERIES: "mistral-large-3"}),

    # --- Qwen -------------------------------------------------------------
    Model("Qwen", "Qwen", 0,
          {A_FORMULA: "qwen", B_SCRIPT: "qwen", C_SERIES: "qwen"}),
    Model("Qwen-3", "Qwen", 1,
          {A_FORMULA: "qwen3", B_SCRIPT: "qwen3", C_SERIES: "qwen3"}),

    # --- DeepSeek ---------------------------------------------------------
    Model("DeepSeek", "DeepSeek", 0,
          {A_FORMULA: "deepseek", B_SCRIPT: "deepseek", C_SERIES: "deepseek"}),
    # Note the swapped column names between datasets -- not a typo here.
    Model("DeepSeek-R1-0528", "DeepSeek", 1,
          {A_FORMULA: "deepseek_r1_0528", B_SCRIPT: "deepseek_r1_0525", C_SERIES: "deepseek_r1_0528"}),

    # --- Meta -------------------------------------------------------------
    Model("Meta", "Meta", 0,
          {A_FORMULA: "meta", B_SCRIPT: "meta", C_SERIES: "meta"}),
    Model("Llama-4-Scout", "Meta", 1,
          {A_FORMULA: "llama_4_scout", B_SCRIPT: "llama_4_scout", C_SERIES: "llama_4_scout"}),

    # --- Other ------------------------------------------------------------
    Model("Cursor-Small", "Cursor", 0,
          {A_FORMULA: "cursor_small", B_SCRIPT: "cursor_small", C_SERIES: "cursor_small"}),
)

# Columns that exist in a dataset but are deliberately not part of the benchmark.
# Retirement must be declared here; it can no longer happen by omission.
RETIRED_COLUMNS: dict[str, dict[str, str]] = {
    B_SCRIPT: {
        "gemini": (
            "Superseded original Gemini run. Gemini is represented in this "
            "dataset by `gemini-thinking`, and Gemini-2.5-Pro by "
            "`gemini-2.5-pro`. Excluded from published Figure 5; Figure 6 "
            "included it by mistake under the label Gemini-2.5-Pro."
        ),
    },
    A_FORMULA: {
        "grok": (
            "Superseded Grok run, excluded from published Figures 3 and 4, which "
            "used `grok_3` for Grok-3. Its cells duplicate `grok_3` and hold "
            "Python code rather than formulae -- script answers that leaked into "
            "the formula dataset."
        ),
        "gemini-thinking": (
            "Superseded Gemini run, excluded from published Figures 3 and 4. In "
            "this dataset Gemini is `gemini`; `gemini-thinking` is the script "
            "dataset's name for it and appears here only as a stray single-variant "
            "column."
        ),
    },
}

# Labels as they appear in the published paper, where these differ from the
# canonical name above. Kept so a parity check can explain a legend difference
# instead of merely reporting one.
PUBLISHED_LABEL_OVERRIDES: dict[tuple[str, str], str] = {
    # Figures 3 and 4 printed the model as "DeepSeek-R1-0525"; Table 1 and
    # Figures 7-10 printed "DeepSeek-R1-0528". Same model.
    (A_FORMULA, "DeepSeek-R1-0528"): "DeepSeek-R1-0525",
}

FAMILY_ORDER: tuple[str, ...] = (
    "OpenAI", "Grok", "Claude", "Gemini", "Mistral", "Qwen", "DeepSeek", "Meta", "Cursor",
)


# --- column naming grammar --------------------------------------------------
# A dataset column is `<model-name><suffix>`. The model name is an atomic
# keyword, so it must be matched WHOLE -- never as a prefix.
#
# Matching by prefix is wrong and was actively harmful: `col.startswith('gpt_4o_')`
# also matches every `gpt_4o_mini_*` column, so ChatGPT-4o's accuracy absorbed
# ChatGPT-4o-Mini's answers. The same collision hit gemini/gemini_2.5_pro,
# mistral/mistral_large2405 and deepseek/deepseek_r1_0528.
#
# Resolution is suffix-first, then longest-name-wins. The second step is what
# makes it correct: several model names themselves end in `_<digit>`
# (`grok_3`, `opus_4`, `chatgpt_5`), so `grok_3_1_eval` must bind to `grok_3`
# and not to `grok` with a leftover `_3_1_eval`.
COLUMN_SUFFIXES: dict[str, tuple[str, ...]] = {
    A_FORMULA: (
        r"",              # the raw answer list
        r"_\d+",          # one extracted formula
        r"_\d+_eval",     # the sequence that formula produces
    ),
    B_SCRIPT: (
        r"",              # the raw list of scripts
    ),
    C_SERIES: (
        r"-formula", r"-formula-eval", r"-formula-correctness",
        r"-formula-ordinal", r"-formula-copy_seq",
        r"-program", r"-program-eval", r"-program-print", r"-program-correctness",
    ),
}


class UnknownColumnError(ValueError):
    """A dataset column matches no registered model and is not declared retired."""


def models_for(dataset: str) -> list[Model]:
    """Registered models present in ``dataset``, in canonical order."""
    return [m for m in MODELS if dataset in m.aliases]


def columns_for(dataset: str) -> list[str]:
    return [m.aliases[dataset] for m in models_for(dataset)]


def by_column(dataset: str, column: str) -> Model | None:
    for m in MODELS:
        if m.aliases.get(dataset) == column:
            return m
    return None


def display_name(dataset: str, column: str, published: bool = False) -> str:
    """Canonical label for a column, or the published label when it differed."""
    model = by_column(dataset, column)
    if model is None:
        raise UnknownColumnError(f"{dataset}: no model registered for column {column!r}")
    if published:
        return PUBLISHED_LABEL_OVERRIDES.get((dataset, model.name), model.name)
    return model.name


def validate_columns(dataset: str, columns: list[str]) -> None:
    """Fail loudly on any model column the registry does not recognise.

    ``columns`` should be the model-bearing columns of the dataset, with
    structural columns such as ``sequence`` and ``Complexity`` already removed.
    """
    known = set(columns_for(dataset))
    retired = set(RETIRED_COLUMNS.get(dataset, {}))
    unknown = [c for c in columns if c not in known and c not in retired]
    if unknown:
        raise UnknownColumnError(
            f"{dataset}: {len(unknown)} unregistered column(s): {sorted(unknown)}.\n"
            "Add each to superarc.registry.MODELS, or declare it in "
            "RETIRED_COLUMNS with the reason. Columns are never dropped silently."
        )
    missing = [c for c in known if c not in columns]
    if missing:
        raise UnknownColumnError(
            f"{dataset}: registered model(s) absent from the data: {sorted(missing)}"
        )


def _candidate_names(dataset: str) -> list[str]:
    """Every name a column may carry in ``dataset``, longest first."""
    names = list(columns_for(dataset)) + list(RETIRED_COLUMNS.get(dataset, {}))
    return sorted(names, key=len, reverse=True)


def resolve_column(dataset: str, column: str) -> tuple[str, str] | None:
    """Split a column into ``(model_column_name, suffix)``.

    Returns ``None`` when the column belongs to no known model. Never matches a
    model name as a prefix of a longer one -- see COLUMN_SUFFIXES.
    """
    import re

    for name in _candidate_names(dataset):          # longest name wins
        for suffix in COLUMN_SUFFIXES.get(dataset, (r"",)):
            if re.fullmatch(re.escape(name) + suffix, column):
                return name, column[len(name):]
    return None


def columns_of(dataset: str, model_column: str, all_columns: list[str]) -> list[str]:
    """All columns belonging to one model, resolved by whole name.

    Use this instead of ``startswith``. Given ``gpt_4o`` it returns the
    ``gpt_4o*`` columns and none of the ``gpt_4o_mini*`` ones.
    """
    out = []
    for column in all_columns:
        resolved = resolve_column(dataset, column)
        if resolved is not None and resolved[0] == model_column:
            out.append(column)
    return out


def prefix_collisions(dataset: str, all_columns: list[str]):
    """Columns that ``startswith`` matching would have stolen from another model.

    The defect this guards against, made visible. Model names are not prefixes,
    they are atomic keywords: ``gpt_4o`` is a different model from
    ``gpt_4o_mini``, but ``column.startswith("gpt_4o")`` is true for both. The
    published figures were built that way, so five models were scored partly on
    a longer-named model's answers.

    Returns one row per model that loses columns, with the columns each would
    have absorbed. An empty frame means no name in this dataset is a prefix of
    another.
    """
    import pandas as pd

    rows = []
    for model in models_for(dataset):
        name = model.column(dataset)
        if name is None:
            continue
        correct = set(columns_of(dataset, name, all_columns))
        by_prefix = {c for c in all_columns if c.startswith(name)}
        stolen = sorted(by_prefix - correct)
        if stolen:
            rows.append({
                "model": name,
                "own columns": len(correct),
                "wrongly absorbed": len(stolen),
                "examples": ", ".join(stolen[:3]) + ("..." if len(stolen) > 3 else ""),
            })
    return pd.DataFrame(rows)


def sorted_by_family(models: list[Model] | None = None) -> list[Model]:
    """Models grouped by family in publication order, oldest first within family."""
    pool = list(MODELS if models is None else models)
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return sorted(pool, key=lambda m: (rank.get(m.family, 99), m.order, m.name))

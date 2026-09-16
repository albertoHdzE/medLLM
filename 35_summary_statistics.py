"""Supplementary Figures 5 and 6. Forwarder onto :mod:`superarc.evolution`.

Kept because it is the path that produced the published figures, and because the
Supplementary Information's evolution panels trace to this name. The analysis
lives in ``superarc/evolution.py``, which owns it.

Running it does what it always did::

    python 35_summary_statistics.py

Output goes to ``superarc.plots_dir()``, which is ``outputs/latest/`` unless
``SUPERARC_PLOTS_DIR`` says otherwise, so a regeneration never writes over the
published artifacts in ``plots/``.
"""

from superarc.evolution import (  # noqa: F401  re-exported for existing callers
    FAMILY_ORDER,
    FORMULA_NAME_MAP,
    SCRIPT_NAME_MAP,
    STACK_ORDER,
    TYPE_COLORS,
    VALID_INSTANCES_COLOR,
    aggregate_data,
    get_family,
    plot_evolution_subset,
    process_data_and_generate_csvs,
    produce,
)

if __name__ == "__main__":
    produce()

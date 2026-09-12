"""SuperARC — reproducible pipeline for the figures and tables of

    "SuperARC: a test for artificial superintelligence based on compressed
     modelling, recursive prediction and problem complexity"
    Nature Communications (2026) 17:4885, doi:10.1038/s41467-026-73289-5

The published results are immutable. New models are appended; existing columns
and previously published values are never rewritten. See
``plans/that-is-a-yes-modular-orbit.md`` for the full design.
"""

__version__ = "0.1.0"

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

# Where a regeneration writes when nothing says otherwise.
#
# This default is load-bearing. The published figures are tracked files in
# ``new_plots/`` and ``plots/highResolution/``, and every producer used to write
# straight into them -- so running any producer plainly, to look at what it does,
# silently overwrote the artifact it was supposed to be compared against. A
# reference you can destroy by accident is not a reference.
#
# ``outputs/`` is git-ignored, so a regeneration leaves the working tree clean and
# ``git status`` stays meaningful. Override with ``SUPERARC_PLOTS_DIR``, which is
# what ``superarc.parity`` and ``superarc.cli`` do.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "latest"


def plots_dir():
    """The directory this run writes figures to. Created if absent."""
    import os
    from pathlib import Path

    override = os.environ.get("SUPERARC_PLOTS_DIR")
    path = Path(override) if override else DEFAULT_OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path

# pybdm imports pkg_resources at module load, which setuptools deprecated in 81
# and removed in 84. We pin setuptools<81 on purpose (see requirements.txt), so
# the warning is expected on every single run and is pure noise. Silenced here,
# once, rather than in each module that touches BDM.
__import__("warnings").filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

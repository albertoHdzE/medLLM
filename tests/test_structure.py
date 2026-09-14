"""Structural guards: one concept, one owner.

A census finds *identical* copies. It misses the ones that drifted, which are the
dangerous ones, because a drifted copy has a different name and a different
signature. So these guards key on a distinctive **body fragment** rather than on
a function name, and they assert the owner's path rather than merely counting.

This is not hypothetical housekeeping. Every defect found in this project came
from a copy that had diverged:

* ``30_multiFormula_experiment.ipynb`` carried five of the formulae-analysis
  functions under the same names, missing the prefix-collision correction;
* ``ascii_to_binary_list`` and ``ascii_to_bits`` were the same function under two
  names -- verified by running them against each other, identical on 8 of 8
  inputs -- which is exactly why a name-based search never found the pair;
* ``31-1_multiScript_experiment.py`` held two conflicting display maps, so the
  same label denoted two different models in two different figures.

Rules these guards follow, so they cannot pass vacuously:

* zero matches is a **failure**, not a pass -- otherwise the guard goes quiet the
  moment someone renames the thing it protects;
* the denominator is printed, so "it passed" is never mistaken for "it looked";
* copies that are known and not yet collapsed are declared below with the reason,
  rather than being absent and therefore invisible.
"""

from __future__ import annotations

import subprocess

import pytest

from superarc import REPO_ROOT

EXCLUDE = ("venv/", ".venv/", "outputs/", ".git/")

# This file necessarily contains every fragment it searches for, so it would
# otherwise report itself as a copy of everything it guards.
SELF = "tests/test_structure.py"


def _search(fragment: str) -> list[str]:
    """Files in the working tree containing a literal fragment.

    Searches the whole repository, not the package being edited -- a radius
    smaller than that is how a mirror gets written.
    """
    args = ["grep", "-rlF", fragment, "--include=*.py", "--include=*.ipynb", "."]
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
    found = [line[2:] if line.startswith("./") else line
             for line in result.stdout.splitlines() if line]
    return sorted(f for f in found
                  if f != SELF and not any(f.startswith(e) for e in EXCLUDE))


def _corpus_size() -> int:
    result = subprocess.run(
        ["bash", "-c",
         "ls *.py *.ipynb superarc/*.py tests/*.py notebooks/*.ipynb 2>/dev/null | wc -l"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return int(result.stdout.strip() or 0)


# concept -> (distinctive body fragment, the one file allowed to contain it)
OWNERS = {
    "pairwise equivalence between a model's own answers":
        ("matching_pairs / total_pairs", "superarc/formulae.py"),
    "known-sequence keyword classification":
        ("'fibonacci', 'prime', 'binary'", "superarc/formulae.py"),
    "Levenshtein dynamic programme":
        ("previous[j - 1] + (ca != cb)", "superarc/timeseries.py"),
    "SuperARC-seq class weights":
        ("CLASS_WEIGHTS = np.array", "superarc/table1.py"),
    "placeholder expectation for missing formulae":
        ("PLACEHOLDER_EXPECTATION = {", "superarc/compression_metrics.py"),
    "suffix-first, longest-name-wins column resolution":
        ("def resolve_column", "superarc/registry.py"),
}

# Copies that exist, are known, and are not collapsed yet. Declared so they are
# visible rather than merely absent. Each names what will collapse it.
#
# Leaving these undeclared is how a census reports three copies where there are
# six -- the guard only inspects what is already complying.
KNOWN_DUPLICATES = {
    "def ascii_to_binary_list": (
        2,
        "owner is superarc/complexity_measures.py; 34-2_S-ARC_ext.py still "
        "carries its own. A third copy lived in superarc/table1.py under the "
        "name `ascii_to_bits` -- found by body fragment, not by name, and "
        "collapsed once running the two showed them identical on 8 of 8 inputs. "
        "The last copy goes with the SuperARC-seq extraction (notebook 05).",
    ),
    "def get_model_display_name": (
        4,
        "owner is superarc/registry.display_name; superarc/formulae.py holds a "
        "thin adapter onto it, while 31-1_multiScript_experiment.py and "
        "34-2_S-ARC_ext.py still carry their own maps -- and 31-1 holds two that "
        "conflict, which is how one label came to denote two different models in "
        "two figures. Collapses with the script extraction (notebook 04).",
    ),
    "def compare_sequences": (
        3,
        "owner is superarc/formulae.py; 31-1_multiScript_experiment.py and its "
        "notebook copy still carry their own. Collapses with notebook 04.",
    ),
    "FAMILY_ORDER": (
        2,
        "superarc/registry.py orders families and derives within-family order "
        "from Model.order; 35_summary_statistics.py carries a dict that spells "
        "both out. They are NOT the same shape, and they disagree -- the local "
        "one says 'Deepseek' where the registry says 'DeepSeek', and omits "
        "'Cursor' entirely. Not collapsed blind, because the within-family "
        "ordering sets the layout of the evolution figures. Collapses with "
        "notebook 06, after the two orderings are diffed elementwise.",
    ),
}


@pytest.mark.parametrize("concept", sorted(OWNERS))
def test_each_concept_has_exactly_one_owner(concept, capsys):
    fragment, owner = OWNERS[concept]
    found = _search(fragment)

    with capsys.disabled():
        print(f"\n  {concept}: {len(found)} of {_corpus_size()} files")

    assert found, (
        f"{concept!r}: fragment {fragment!r} found nowhere. Either the owner was "
        f"renamed and this guard has gone silent, or it was deleted."
    )
    assert found == [owner], (
        f"{concept!r} should live only in {owner}, found in {found}"
    )


@pytest.mark.parametrize("fragment", sorted(KNOWN_DUPLICATES))
def test_known_duplicates_have_not_multiplied(fragment):
    """Declared copies may be collapsed, never added to."""
    expected, reason = KNOWN_DUPLICATES[fragment]
    found = _search(fragment)
    assert found, f"{fragment!r} found nowhere; the declaration is stale"
    assert len(found) <= expected, (
        f"{fragment!r} now in {len(found)} files, was {expected}: {found}\n{reason}"
    )


def test_the_guard_fires_when_a_copy_is_planted(tmp_path):
    """A guard nobody has seen fail is decoration.

    Plants a copy of a guarded fragment inside the repository, confirms the
    search finds two files instead of one, and removes it again.
    """
    fragment, owner = OWNERS["pairwise equivalence between a model's own answers"]
    assert _search(fragment) == [owner], "preconditions not met"

    planted = REPO_ROOT / "_guard_probe.py"
    try:
        planted.write_text(f"# planted by the structural guard test\nx = '{fragment}'\n")
        found = _search(fragment)
        assert len(found) == 2, f"guard did not see the planted copy: {found}"
        assert "_guard_probe.py" in found
    finally:
        planted.unlink(missing_ok=True)

    assert _search(fragment) == [owner], "probe not cleaned up"


def test_the_dependency_points_one_way():
    """Scripts load the package; the package never loads a script.

    Checked on executable statements, not on mentions -- the modules refer to
    the producer scripts by name in their docstrings, which is documentation and
    is the point. An earlier version of this test matched those mentions and
    failed on its own explanatory prose.
    """
    import ast

    offenders = []
    for module in sorted((REPO_ROOT / "superarc").glob("*.py")):
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(ch.isdigit() for ch in name.split(".")[0][:2]):
                    offenders.append(f"{module.name} imports {name}")

    assert not offenders, "the package must not import a producer script: " + "; ".join(offenders)

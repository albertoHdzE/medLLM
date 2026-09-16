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
    "the SuperARC-seq score itself":
        ("rho[:3] * delta * CLASS_WEIGHTS", "superarc/table1.py"),
    "the four answer classes":
        ("correct & ordinal", "superarc/table1.py"),
    "string to the bit array BDM measures":
        ("format(ord(char), '08b')", "superarc/complexity_measures.py"),
    "placeholder expectation for missing formulae":
        ("PLACEHOLDER_EXPECTATION = {", "superarc/compression_metrics.py"),
    "suffix-first, longest-name-wins column resolution":
        ("def resolve_column", "superarc/registry.py"),
    "the polyglot language summary chain":
        ('normalized_compressed_{}.csv', "superarc/languages.py"),
    "the ordering of model families":
        ("FAMILY_ORDER: tuple[str, ...] = (", "superarc/registry.py"),
    "the family-evolution figures":
        ("def plot_evolution_subset", "superarc/evolution.py"),
    "pairwise equivalence between a model's own programs":
        ("matching_comparisons / total_comparisons", "superarc/scripts.py"),
    "generated-program type classification":
        ("'fibonacci', 'fib', 'prime', 'lucas'", "superarc/scripts.py"),
    "splitting a model's answer into one program per column":
        ("def split_model_scripts", "superarc/scripts.py"),
    "normalising a program's printed output":
        ("def clean_result", "superarc/scripts.py"),
}

# Copies that exist, are known, and are not collapsed yet. Declared so they are
# visible rather than merely absent. Each names what will collapse it.
#
# Leaving these undeclared is how a census reports three copies where there are
# six -- the guard only inspects what is already complying.
KNOWN_DUPLICATES = {
    "def get_model_display_name": (
        3,
        "owner is superarc/registry.display_name; superarc/formulae.py, "
        "superarc/scripts.py and superarc/superarc_seq.py hold thin adapters "
        "onto it, one per dataset, because each dataset names the same model "
        "differently and the adapter is where that dataset is named. Every "
        "hand-written map is gone: 34-2_S-ARC_ext.py's agreed with the registry "
        "on all 28 entries and was deleted on that evidence, and "
        "31-1_multiScript_experiment.py held TWO that conflicted -- Figure 5 "
        "mapped `gemini-2.5-pro` to Gemini-2.5-Pro, Figure 6 mapped `gemini` to "
        "it.",
    ),
    # Keyed on the call rather than on the bare `sets_ordering='minimize gaps'`
    # option: that string is discussed in notebooks/08's prose, and a guard that
    # counts prose as a copy trains its reader to raise the expected number until
    # it stops guarding anything.
    "supervenn(sets, labels , sets_ordering=": (
        2,
        "owner is superarc/languages.plot_sequence_overlap; the second is "
        "24_PLOTS_paper.ipynb, kept as the path that produced Supplementary "
        "Figures 2-4. It is not a live second owner: it reads one input through "
        "/Users/beto/Documents/Projects/medLLM/, an absolute path on another "
        "machine, so it cannot run here at all -- which is why those five panels "
        "had no gate until superarc/languages.py was written. Kept rather than "
        "deleted because the published figures trace to that name. The lab's "
        "copy (AlgoDynLab/SuperintelligenceTest, 05_PLOTS_paper.ipynb) carries "
        "the identical defect.",
    ),
    "def compare_sequences": (
        2,
        "NOT a duplicate: superarc/formulae.py counts a prefix of the target as "
        "a match, superarc/scripts.py demands the whole sequence. Two published "
        "definitions of 'right', one per experiment. Declared here because a "
        "census counts them as two copies of one name, and because collapsing "
        "them would silently move one of the two experiments. Pinned in "
        "tests/test_scripts.py::test_compare_sequences.",
    ),
    # COLLAPSED 2026-09-16. Was: two FAMILY_ORDERs, one in superarc/registry.py
    # and one hand-written in 35_summary_statistics.py, which disagreed. The
    # elementwise diff this entry demanded before collapsing is now
    # tests/test_evolution.py::test_the_two_orderings_agree_once_the_phantoms_are_removed,
    # kept as an assertion rather than as a note: six of the eight families were
    # character-for-character identical, and the three differences were two
    # phantom names ('Grok', 'DeepSeek-R1-0525' -- in no dataset, and shifting
    # every later rank uniformly, so the drawn order was unchanged) and a family
    # key spelt 'Deepseek'. The spelling turned out to be load-bearing: it is
    # what the Supplementary Information PRINTS, so it survives as
    # evolution.PUBLISHED_FAMILY_LABEL, applied at the point of drawing.
    #
    # The hand-written dict itself survives once, in that test, as the record of
    # what was collapsed. A second occurrence would mean it grew back.
    "'Deepseek': [": (
        1,
        "the pre-collapse ordering, kept in tests/test_evolution.py so the diff "
        "that licensed the collapse is checked rather than remembered. The live "
        "ordering is derived from superarc/registry.py.",
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

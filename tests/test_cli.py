"""Checks for the one-command regeneration.

These test the manifest and the safety property, not the figures -- the figures
are ``superarc.parity``'s job. The safety property is the important one: a
regeneration must write only into its output directory, because the alternative
is silently overwriting the published artifacts it is meant to be compared
against.
"""

from __future__ import annotations

import json

import pytest

from superarc import REPO_ROOT
from superarc.cli import (
    MANIFEST_NAME,
    git_revision,
    input_hashes,
    models_present,
    package_versions,
    write_manifest,
)


def test_manifest_records_every_input_dataset():
    from superarc.baseline import DATASETS

    hashes = input_hashes()
    assert len(hashes) == len(DATASETS)
    assert all(len(digest) == 64 for digest in hashes.values())


def test_manifest_records_the_models_each_dataset_carries():
    models = models_present()
    assert set(models) == {"A_formula", "B_script", "C_series"}
    for dataset, names in models.items():
        assert len(names) > 20, dataset
        assert len(names) == len(set(names)), f"{dataset} has duplicate models"


def test_setuptools_is_pinned_below_the_version_that_removed_pkg_resources():
    """pybdm imports pkg_resources at module load.

    setuptools 81 deprecated it and 84 removed it. On a version that removed it,
    BDM fails inside joblib workers as a BrokenProcessPool -- an error that names
    nothing relevant. Worth failing here instead.
    """
    version = package_versions()["setuptools"]
    assert version != "not installed"
    assert int(version.split(".")[0]) < 81, (
        f"setuptools {version} breaks pybdm; pin setuptools<81"
    )


def test_git_revision_is_recorded():
    revision = git_revision()
    assert set(revision) == {"commit", "branch", "dirty"}
    assert isinstance(revision["dirty"], bool)


def test_manifest_is_valid_json_and_lists_the_outputs(tmp_path):
    (tmp_path / "figure03-up.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "ranking_table.csv").write_text("Model,phi\n")

    path = write_manifest(tmp_path)
    manifest = json.loads(path.read_text())

    assert manifest["outputs"] == ["figure03-up.pdf", "ranking_table.csv"]
    assert MANIFEST_NAME not in manifest["outputs"]
    assert manifest["python"]
    assert manifest["inputs"]


@pytest.mark.parametrize(
    "path",
    ["plots/highResolution/figure01.pdf", "new_plots/figure05.png"],
)
def test_published_artifacts_are_where_the_gate_expects_them(path):
    """If these move, parity silently stops comparing anything."""
    assert (REPO_ROOT / path).exists(), path

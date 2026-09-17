"""No secret in the working tree, and a working way to supply one.

Two Nixtla keys were pasted into ``processing_answers.py`` and are still readable
in this repository's public git history. The parser and accessors below are the
replacement path; the scan at the bottom is the part that matters, because a
convention nobody checks is how the first one got in.

What these do NOT do is make the exposure go away. The keys are published and
must be revoked. See :mod:`superarc.credentials`.
"""

from __future__ import annotations

import subprocess

import pytest

from superarc import REPO_ROOT, credentials as C


# ---- the parser ---------------------------------------------------------------


def test_parses_the_forms_people_actually_write():
    parsed = C.parse_env(
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        "  SPACED  =  padded  \n"
        'DOUBLE="quoted value"\n'
        "SINGLE='quoted value'\n"
        "export EXPORTED=exported\n"
        "EMPTY=\n"
        "no_equals_sign_here\n"
    )
    assert parsed == {
        "PLAIN": "value",
        "SPACED": "padded",
        "DOUBLE": "quoted value",
        "SINGLE": "quoted value",
        "EXPORTED": "exported",
        "EMPTY": "",
    }


def test_a_malformed_env_cannot_stop_a_figure_regenerating():
    """No figure needs a credential, so a broken .env must not raise."""
    assert C.parse_env("]]] not an env file [[[\n\x00\n") == {}


def test_the_shell_wins_over_the_file(tmp_path, monkeypatch):
    """CI supplies the secret as an environment variable and has no .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("SUPERARC_PROBE=from_file\n")

    monkeypatch.setenv("SUPERARC_PROBE", "from_shell")
    C.load_env(env_file)
    assert C.parse_env(env_file.read_text())["SUPERARC_PROBE"] == "from_file"
    import os
    assert os.environ["SUPERARC_PROBE"] == "from_shell"

    C.load_env(env_file, override=True)
    assert os.environ["SUPERARC_PROBE"] == "from_file"


def test_a_missing_env_file_is_not_an_error(tmp_path):
    assert C.load_env(tmp_path / "nope.env") == {}


# ---- the accessors -------------------------------------------------------------


def test_no_key_configured_is_a_supported_state(monkeypatch):
    """The state every reproduction of the paper runs in."""
    monkeypatch.setattr(C, "ENV_FILE", REPO_ROOT / "does-not-exist")
    monkeypatch.setattr(C, "_loaded", True)
    monkeypatch.delenv(C.NIXTLA_KEY_VAR, raising=False)

    assert C.nixtla_api_key() is None
    assert C.nixtla_client(required=False) is None


def test_asking_for_a_missing_key_says_what_to_do(monkeypatch):
    monkeypatch.setattr(C, "_loaded", True)
    monkeypatch.delenv(C.NIXTLA_KEY_VAR, raising=False)

    with pytest.raises(C.MissingCredential) as excinfo:
        C.nixtla_api_key(required=True)
    message = str(excinfo.value)
    assert ".env" in message
    assert "parity" in message, "must say that no key is needed to reproduce"


def test_the_audit_never_prints_the_secret(monkeypatch):
    monkeypatch.setattr(C, "_loaded", True)
    monkeypatch.setenv(C.NIXTLA_KEY_VAR, "super-secret-value-0123456789")

    report = C.audit()
    assert "super-secret-value-0123456789" not in repr(report)
    assert report["length"] == 29
    assert len(report["sha256"]) == 12


def test_the_closed_experiment_still_imports_without_a_key(monkeypatch):
    """Importing must not reach the network, or parity would need credentials."""
    monkeypatch.delenv(C.NIXTLA_KEY_VAR, raising=False)
    import processing_answers as PA

    assert PA.pipeline is None
    assert PA.get_timegpt() is None


# ---- the part that matters ------------------------------------------------------


# The shapes a pasted credential takes. Kept as patterns rather than as the two
# known keys, so this catches the NEXT one rather than only the last two.
#
# The character class is written without backslash escapes and with `-` last, on
# purpose. POSIX bracket expressions do not honour backslash escapes: an earlier
# version of this pattern used `[A-Za-z0-9_\-\.\+/=]`, which grep read as
# containing a literal backslash and a range starting at it, and it matched
# nothing at all. The planted-secret test below is what caught that -- until it
# did, the scan above was passing vacuously.
SECRET_PATTERN = (
    r"(api[_-]?key|apikey|token|secret|password|passwd|bearer)"
    r"[\"']?[[:space:]]*[:=][[:space:]]*[\"'][A-Za-z0-9_./+=-]{16,}[\"']"
)


def _grep(pattern: str) -> list[str]:
    result = subprocess.run(
        ["grep", "-rlEi", pattern, "--include=*.py", "--include=*.ipynb",
         "--include=*.toml", "--include=*.cfg", "--include=*.json",
         "--include=*.sh", "--include=*.yml", "--include=*.yaml", "."],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    exclude = ("venv/", ".venv/", "outputs/", ".git/", "tests/test_credentials.py")
    found = [f[2:] if f.startswith("./") else f
             for f in result.stdout.splitlines() if f]
    return sorted(f for f in found if not any(f.startswith(e) for e in exclude))


def test_no_credential_is_assigned_anywhere_in_the_working_tree():
    found = _grep(SECRET_PATTERN)
    assert not found, (
        f"a credential-shaped literal is assigned in: {found}\n"
        f"Put it in .env instead (see .env.example). A secret committed here is "
        f"published permanently -- the two in this file's history still are."
    )


def test_the_scan_fires_when_a_secret_is_planted(tmp_path):
    """A search that finds nothing must be tested against a known hit.

    This repository has already been bitten once by a history scan whose loop
    could not see `git` on its PATH: every lookup failed silently and no output
    was read as no problem. A secret scanner that has never matched anything is
    the same failure waiting to happen.
    """
    planted = REPO_ROOT / "_secret_probe.py"
    try:
        planted.write_text('api_key = "nixak-0123456789abcdefghijklmnop"\n')
        found = _grep(SECRET_PATTERN)
        assert "_secret_probe.py" in found, f"scanner missed a planted secret: {found}"
    finally:
        planted.unlink(missing_ok=True)
    assert not _grep(SECRET_PATTERN), "probe not cleaned up"


def test_dotenv_is_ignored_and_the_template_is_not():
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".env"], cwd=REPO_ROOT).returncode == 0
    assert ignored, ".env must be git-ignored"

    template_ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".env.example"], cwd=REPO_ROOT).returncode == 0
    assert not template_ignored, ".env.example must be committed"
    assert C.TEMPLATE_FILE.exists()


def test_the_template_holds_no_value():
    """A template with a key in it is just a key."""
    for key, value in C.parse_env(C.TEMPLATE_FILE.read_text()).items():
        assert value == "", f"{key} has a value in .env.example"


def test_dotenv_is_not_tracked():
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        cwd=REPO_ROOT, capture_output=True, text=True).returncode == 0
    assert not tracked, ".env is tracked by git; it must not be"

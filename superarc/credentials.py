"""Credentials, read from the environment or from a git-ignored ``.env``.

No published result needs one. The forecasting experiment is closed, its outputs
are the committed ``timeGPT_*.csv`` and ``chronos_*.csv``, and
``python -m superarc.parity`` regenerates every figure with both back ends set to
``None``. This module exists so that *re-running* that experiment is possible
without a secret ever entering a source file again.

**It does not undo the exposure.** Two Nixtla keys are in this repository's git
history, and that history is public: ``origin/master:processing_answers.py``
still carries both as literal strings, and the 86-character one is also on the
``main`` branch of ``AlgoDynLab/SuperintelligenceTest``. Moving a secret out of
the working tree changes nothing about a secret that has already been published.
Both keys must be revoked at ``dashboard.nixtla.io``; this module is what the
replacements get loaded through, not a substitute for revoking them.

A deliberate non-dependency: the parser below is about thirty lines rather than
``python-dotenv``. The environment for this project is pinned and rebuilt from a
lockfile so that published figures stay reproducible, and adding a package to
that lockfile to read ``KEY=value`` is a poor trade.

Usage::

    from superarc.credentials import nixtla_api_key
    key = nixtla_api_key()          # None if unset
    key = nixtla_api_key(required=True)   # raises with instructions if unset
"""

from __future__ import annotations

import os
from pathlib import Path

from . import REPO_ROOT

ENV_FILE = REPO_ROOT / ".env"
TEMPLATE_FILE = REPO_ROOT / ".env.example"

NIXTLA_KEY_VAR = "NIXTLA_API_KEY"

_loaded = False


def parse_env(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Blank lines and ``#`` comments are skipped.

    Accepts an optional ``export`` prefix and single or double quotes around the
    value, because those are what people actually write. A line without ``=`` is
    ignored rather than raising: a malformed ``.env`` should not be able to stop
    a figure from regenerating, since no figure needs one.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Load ``.env`` into ``os.environ``. Returns what it set.

    ``override`` is False by default, so a variable already exported in the shell
    wins over the file. That ordering matters for CI, where the secret arrives as
    an environment variable and no ``.env`` exists at all.
    """
    path = ENV_FILE if path is None else Path(path)
    if not path.exists():
        return {}

    applied = {}
    for key, value in parse_env(path.read_text()).items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        load_env()
        _loaded = True


class MissingCredential(RuntimeError):
    """Raised when a back end is asked for and no key is configured."""


def nixtla_api_key(required: bool = False) -> str | None:
    """The Nixtla key, from the environment or ``.env``. ``None`` if unset."""
    _ensure_loaded()
    key = os.environ.get(NIXTLA_KEY_VAR) or None
    if key is None and required:
        raise MissingCredential(
            f"{NIXTLA_KEY_VAR} is not set.\n"
            f"\n"
            f"Nothing in the paper needs it: the forecasting experiment is closed, "
            f"its results are the committed timeGPT_*.csv, and "
            f"`python -m superarc.parity` regenerates every figure without it.\n"
            f"\n"
            f"To re-run the forecasts, copy {TEMPLATE_FILE.name} to .env and put a "
            f"key in it, or export {NIXTLA_KEY_VAR} in your shell. .env is "
            f"git-ignored. Do not paste a key into a source file: the two that "
            f"were pasted into this one are public in the git history and are "
            f"being rotated."
        )
    return key


def nixtla_client(required: bool = True):
    """Build a ``NixtlaClient``, or return ``None`` when nothing is configured.

    Kept here rather than in ``processing_answers`` so that the one place a key
    is turned into a client is the same place the key is read.
    """
    key = nixtla_api_key(required=required)
    if key is None:
        return None
    from nixtla import NixtlaClient

    return NixtlaClient(api_key=key)


def audit() -> dict[str, object]:
    """What is configured, without printing any secret.

    A fingerprint rather than a value, so this can be shown in a notebook or
    pasted into an issue. The length and the hash are enough to tell two keys
    apart and to check one against a record of which key was revoked.
    """
    import hashlib

    _ensure_loaded()
    key = os.environ.get(NIXTLA_KEY_VAR) or None
    return {
        ".env present": ENV_FILE.exists(),
        ".env.example present": TEMPLATE_FILE.exists(),
        f"{NIXTLA_KEY_VAR} set": key is not None,
        "length": len(key) if key else None,
        "sha256": hashlib.sha256(key.encode()).hexdigest()[:12] if key else None,
    }

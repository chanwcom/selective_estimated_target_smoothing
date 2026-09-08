"""Machine-specific paths, read from the environment.

The values are defined in `config.local.sh` -- one gitignored copy per
machine -- and exported by `source set_config.sh`. Keeping them out of the
tracked sources is what lets a fresh `git clone` elsewhere work by editing
one file, instead of hunting absolute paths through every script.

See `config.local.sh.example` for the variables and what they mean.
"""

# pylint: disable=import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

import os


def _require(name: str) -> str:
    """Returns the environment variable `name`, or exits explaining how."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. Run `source set_config.sh` first; if that "
            f"reports a missing config.local.sh, copy it from "
            f"config.local.sh.example and edit it for this machine "
            f"(see README, Setup).")
    return value


CWK_HOME = _require("CWK_HOME")
DB_TOP_DIR = _require("DB_TOP_DIR")
CHECKPOINT_TOP_DIR = _require("CHECKPOINT_TOP_DIR")

# SentencePiece vocabularies are checked into the CWK repo, not this one.
RESOURCE_TOP_DIR = os.path.join(CWK_HOME, "resources", "spm")

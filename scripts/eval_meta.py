"""Run-identity metadata for evaluation artifacts.

WHY THIS EXISTS (2026-09-16)
----------------------------
A core small-paper claim was once derived from an output filename that was
*assumed* to identify a model but did not:

    eval_distilled_d0_s42.json      -> weights = shoot_bc_asap_distilled.pth
    eval_2x2_base_geoOld_s42.json   -> weights = shoot_bc_asap_distilled.pth
                                       (identical file, different name)
    eval_geomA_d0_s42.json          -> weights = shoot_bc_asap_geomA.pth

Reading the filenames suggested a BC-vs-SPC comparison; reading the weights
showed a same-model / different-geometry pair. The conclusion was wrong for
weeks. Therefore: **every evaluation JSON must carry its own identity block.**
Never infer model identity, geometry, or difficulty from the output path.

CONTRACT
--------
Every eval script writes a top-level ``run_meta`` object built by
:func:`build_run_meta`, containing at minimum:

    model_id            short stable identifier of the policy
    checkpoint          path to the weights actually loaded
    checkpoint_sha256   content hash (kills 'same filename, new weights')
    seed_range          [first_seed, last_seed] inclusive
    geometry            initial-geometry descriptor, e.g. "bias U(0,60)"
    difficulty          difficulty_level used
    action_mode         "argmax" or "sample"
    git_head            repo revision of the code that produced it
    script              script that produced it
    timestamp_utc       ISO-8601 UTC
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import subprocess

__all__ = [
    "sha256_file",
    "git_head",
    "geometry_label",
    "build_run_meta",
]

# Heading bias is drawn as U(bias_min, bias_max) with bias_max = 60 fixed by
# the task definition; only the lower bound is configurable (see commit
# fb48155: bias_min 30.0 -> 0.0).
_BIAS_MAX = 60.0


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Content hash of a file, or 'MISSING' if it cannot be read.

    Weights are ~0.4 MB, so a full read is cheap and gives a strong identity
    guarantee: two runs that loaded different weights can never collide.
    """
    if not path or not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
    except OSError:
        return "UNREADABLE"
    return h.hexdigest()


def git_head(repo_root: str, short: bool = True) -> str:
    """Current HEAD of the repo, or 'UNKNOWN' outside a git work tree."""
    try:
        # 'git rev-parse --short' alone is not valid: --short needs a revision
        # to shorten, so HEAD must always be passed explicitly.
        args = ["git", "-C", repo_root, "rev-parse"]
        args += ["--short", "HEAD"] if short else ["HEAD"]
        out = subprocess.run(args, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "UNKNOWN"


def geometry_label(min_heading_bias_deg=None) -> str:
    """Human-readable initial-geometry descriptor.

    None means 'whatever the task default is', which since fb48155 is U(0,60).
    """
    lo = 0.0 if min_heading_bias_deg is None else float(min_heading_bias_deg)
    return f"heading_bias U({lo:g},{_BIAS_MAX:g})"


def build_run_meta(model_id: str,
                   checkpoint: str,
                   first_seed: int,
                   episodes: int,
                   difficulty: float,
                   min_heading_bias_deg=None,
                   action_mode: str = "argmax",
                   script: str = "",
                   repo_root: str = "",
                   extra: dict | None = None) -> dict:
    """Build the mandatory identity block for an evaluation artifact.

    ``model_id`` should be an explicit human-chosen identifier passed on the
    command line (e.g. 'bc_round1', 'spc_distilled') rather than derived from
    the checkpoint filename -- deriving it reintroduces the very coupling this
    module exists to break. Scripts fall back to the checkpoint stem only when
    the caller gives nothing.
    """
    if not model_id:
        model_id = os.path.splitext(os.path.basename(checkpoint or "unknown"))[0]
    if not repo_root:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    meta = {
        "model_id": model_id,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed_range": [int(first_seed), int(first_seed) + int(episodes) - 1],
        "n_episodes": int(episodes),
        "geometry": geometry_label(min_heading_bias_deg),
        "min_heading_bias_deg": (None if min_heading_bias_deg is None
                                 else float(min_heading_bias_deg)),
        "difficulty": float(difficulty),
        "action_mode": action_mode,
        "git_head": git_head(repo_root),
        "script": script or os.path.basename(
            __import__("sys").argv[0] if __import__("sys").argv else "unknown"),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        meta.update(extra)
    return meta

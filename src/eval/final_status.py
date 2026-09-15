"""Evaluator-side final §3.4 status (Lead decision, reports/decisions.md 2026-08-28).

The generator can only see *why* decoding stopped, so its label is provisional
(``gen_status``): ``stop_reason == eos`` + valid audio becomes ``complete`` even
when the model read two thirds of the text and stopped.  The final PLAN.md §3.4
status is assigned here, from the content metrics, with thresholds frozen in
``configs/eval.yaml``:

    gen_status != complete           -> passed through unchanged
    complete + no usable audio       -> empty_or_invalid_audio
    complete + EndCoverage <  0.95   -> early_eos
    complete + EndCoverage >= 0.95 and (WER-floor >= 0.30 or loop) -> degraded
    complete + EndCoverage >= 0.95 and neither                     -> complete

§3.4 has no ``loop`` status on purpose: a looped output that stopped by EOS is
``degraded``; a looped output that hit the watchdog is ``loop_cap`` and that
label comes from the generator untouched (PLAN.md §3.3).

Nothing here reads the file system or the model: it is a pure function of the
per-item metrics, so every branch is unit-testable without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "KNOWN_STATUSES",
    "EVAL_CONFIG_PATH",
    "FinalStatusError",
    "load_eval_config",
    "gen_status_of",
    "loop_verdict",
    "assign_final_status",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "eval.yaml"

# PLAN.md §3.4, mutually exclusive
KNOWN_STATUSES = (
    "complete",
    "degraded",
    "early_eos",
    "loop_cap",
    "hard_input_limit",
    "context_limit",
    "oom",
    "timeout",
    "empty_or_invalid_audio",
    "infrastructure_error",
)

_REQUIRED_KEYS = (
    "refines_gen_status",
    "require_stop_reason_for_complete",
    "end_coverage_complete_min",
    "wer_minus_floor_degraded_min",
    "wer_criterion_when_floor_missing",
    "no_usable_audio_status",
    "loop",
)
_REQUIRED_LOOP_KEYS = ("ngram_range", "excess_repetition_rate_min", "ngram_run_min", "combine")


class FinalStatusError(RuntimeError):
    """The frozen status configuration is missing, malformed or contradicts §3.4."""


def load_eval_config(path: str | Path | None = None) -> dict:
    """Read and validate ``configs/eval.yaml``.

    A missing file or a missing threshold is an error: the status rule must never
    fall back to a hard-coded default that nobody pre-registered.
    """
    import yaml

    p = Path(path) if path else EVAL_CONFIG_PATH
    if not p.exists():
        raise FinalStatusError(
            f"frozen evaluator config not found: {p} (PLAN.md §3.5: thresholds are "
            "pre-registered in configs/eval.yaml, never defaulted in code)"
        )
    with open(p, "rt", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "final_status" not in cfg:
        raise FinalStatusError(f"{p}: no `final_status` block")
    fs = cfg["final_status"]
    missing = [k for k in _REQUIRED_KEYS if k not in fs]
    if missing:
        raise FinalStatusError(f"{p}: final_status is missing {missing}")
    loop_missing = [k for k in _REQUIRED_LOOP_KEYS if k not in fs["loop"]]
    if loop_missing:
        raise FinalStatusError(f"{p}: final_status.loop is missing {loop_missing}")
    if fs["no_usable_audio_status"] not in KNOWN_STATUSES:
        raise FinalStatusError(
            f"{p}: no_usable_audio_status={fs['no_usable_audio_status']!r} is not a "
            f"PLAN.md §3.4 status {KNOWN_STATUSES}"
        )
    if str(fs["loop"]["combine"]).lower() != "or":
        raise FinalStatusError(f"{p}: only `combine: or` is implemented for the loop rule")
    if fs["wer_criterion_when_floor_missing"] != "skip":
        raise FinalStatusError(
            f"{p}: wer_criterion_when_floor_missing="
            f"{fs['wer_criterion_when_floor_missing']!r}; only 'skip' is implemented "
            "(borrowing a floor from another text would invent a number)"
        )
    for k in ("end_coverage_complete_min", "wer_minus_floor_degraded_min"):
        fs[k] = float(fs[k])
    fs["loop"]["excess_repetition_rate_min"] = float(fs["loop"]["excess_repetition_rate_min"])
    fs["loop"]["ngram_run_min"] = int(fs["loop"]["ngram_run_min"])
    cfg.setdefault("wer_valid_denominator", "gen_status")
    cfg["__path__"] = str(p)
    return cfg


def gen_status_of(run: dict) -> str:
    """The generator's provisional status of one run-manifest row.

    Accepts both spellings: the current adapters write ``status`` (§10); once the
    generator is updated to write ``gen_status`` explicitly, that field wins.  If
    a row carries both and they disagree, the row is ambiguous and that is an
    error, not a preference.
    """
    has_gen = "gen_status" in run and run["gen_status"] is not None
    has_status = "status" in run and run["status"] is not None
    if has_gen and has_status and run["gen_status"] != run["status"]:
        raise FinalStatusError(
            f"run {run.get('run_id')!r} carries gen_status={run['gen_status']!r} and "
            f"status={run['status']!r}; the run manifest must not contain two "
            "different generation labels (PLAN.md §10)"
        )
    if has_gen:
        return str(run["gen_status"])
    if has_status:
        return str(run["status"])
    raise FinalStatusError(f"run {run.get('run_id')!r} has neither gen_status nor status")


def loop_verdict(
    excess_repetition_rate: float | None,
    max_ngram_run: int | None,
    cfg: dict,
) -> tuple[bool, str | None]:
    """Frozen loop rule: dispersed excess repetition OR a back-to-back run.

    Returns ``(flag, human-readable reason)``; the reason names the criterion and
    the two numbers, so a flagged item can be checked by hand.
    """
    loop = cfg["final_status"]["loop"]
    reasons: list[str] = []
    rate_min = loop["excess_repetition_rate_min"]
    run_min = loop["ngram_run_min"]
    if excess_repetition_rate is not None and excess_repetition_rate >= rate_min:
        reasons.append(f"excess_repetition_rate={excess_repetition_rate:.4f}>={rate_min:g}")
    if max_ngram_run is not None and max_ngram_run >= run_min:
        reasons.append(f"max_ngram_run={int(max_ngram_run)}>={run_min:d}")
    return (bool(reasons), "; ".join(reasons) if reasons else None)


def assign_final_status(
    *,
    gen_status: str,
    stop_reason: str | None,
    usable_audio: bool,
    end_coverage: float | None,
    wer_minus_floor: float | None,
    floor_available: bool,
    excess_repetition_rate: float | None,
    max_ngram_run: int | None,
    cfg: dict,
) -> dict[str, Any]:
    """Assign the final PLAN.md §3.4 status of one evaluated run.

    ``usable_audio`` means: the WAV exists, the ASR read it and the transcription
    is not empty.  Everything else is a content metric of that transcription.
    """
    fs = cfg["final_status"]
    loop_flag, loop_reason = loop_verdict(excess_repetition_rate, max_ngram_run, cfg)
    out: dict[str, Any] = {
        "gen_status": gen_status,
        "loop_flag": loop_flag,
        "loop_reason": loop_reason,
        "final_status_note": None,
    }

    if gen_status not in KNOWN_STATUSES:
        raise FinalStatusError(
            f"gen_status={gen_status!r} is not one of the PLAN.md §3.4 statuses {KNOWN_STATUSES}"
        )

    if gen_status not in tuple(fs["refines_gen_status"]):
        out["status"] = gen_status
        out["final_status_reason"] = f"gen_status={gen_status} passed through unchanged (§3.4)"
        return out

    required_stop = fs["require_stop_reason_for_complete"]
    if required_stop and stop_reason != required_stop:
        raise FinalStatusError(
            f"gen_status={gen_status!r} with stop_reason={stop_reason!r}: a provisional "
            f"`complete` must come from stop_reason={required_stop!r} (PLAN.md §10)"
        )

    if not usable_audio:
        out["status"] = fs["no_usable_audio_status"]
        out["final_status_reason"] = (
            f"gen_status=complete but no usable audio -> {fs['no_usable_audio_status']}"
        )
        return out

    cov_min = fs["end_coverage_complete_min"]
    cov = end_coverage
    if cov is None:
        raise FinalStatusError(
            "end_coverage is required to finalize a `complete` run; it is missing"
        )
    if cov < cov_min:
        out["status"] = "early_eos"
        out["final_status_reason"] = f"eos with EndCoverage={cov:.4f}<{cov_min:g}"
        return out

    margin = fs["wer_minus_floor_degraded_min"]
    wer_bad = False
    if wer_minus_floor is not None and floor_available:
        wer_bad = wer_minus_floor >= margin
    elif not floor_available:
        out["final_status_note"] = "no_floor_wer_criterion_skipped"

    if wer_bad or loop_flag:
        why = []
        if wer_bad:
            why.append(f"wer_minus_floor={wer_minus_floor:.4f}>={margin:g}")
        if loop_flag:
            why.append(f"loop({loop_reason})")
        out["status"] = "degraded"
        out["final_status_reason"] = f"eos with EndCoverage={cov:.4f}>={cov_min:g} but " + " and ".join(why)
        return out

    out["status"] = "complete"
    out["final_status_reason"] = (
        f"eos with EndCoverage={cov:.4f}>={cov_min:g}, no loop, "
        + (f"wer_minus_floor={wer_minus_floor:.4f}<{margin:g}"
           if (wer_minus_floor is not None and floor_available) else "no floor available")
    )
    return out

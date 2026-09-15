"""Global monotonic word alignment between reference and hypothesis (PLAN.md §9.3).

Thin, explicit wrapper over ``jiwer.process_words`` so that every downstream
metric (coverage, EndCoverage, longest deletion run, tail deletion rate,
rolling windows) reads the same op list instead of re-deriving an alignment.

An op is a dict::

    {"op": "equal"|"substitute"|"delete"|"insert",
     "ref_start": int, "ref_end": int,      # half-open, indices into ref words
     "hyp_start": int, "hyp_end": int}      # half-open, indices into hyp words

The op list is monotonic and covers every reference word exactly once and every
hypothesis word exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["Alignment", "align_words", "OPS"]

OPS = ("equal", "substitute", "delete", "insert")


@dataclass
class Alignment:
    """Result of a global monotonic word alignment."""

    reference: list[str]
    hypothesis: list[str]
    ops: list[dict] = field(default_factory=list)
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    # -- derived views ------------------------------------------------------

    @property
    def n_ref(self) -> int:
        return len(self.reference)

    @property
    def n_hyp(self) -> int:
        return len(self.hypothesis)

    def ref_aligned_mask(self) -> list[bool]:
        """True for reference words matched by a hit or a substitution."""
        mask = [False] * self.n_ref
        for op in self.ops:
            if op["op"] in ("equal", "substitute"):
                for i in range(op["ref_start"], op["ref_end"]):
                    mask[i] = True
        return mask

    def ref_deleted_mask(self) -> list[bool]:
        mask = [False] * self.n_ref
        for op in self.ops:
            if op["op"] == "delete":
                for i in range(op["ref_start"], op["ref_end"]):
                    mask[i] = True
        return mask

    def ref_to_hyp_index(self) -> list[int | None]:
        """For each reference word: index of the aligned hypothesis word, or None."""
        out: list[int | None] = [None] * self.n_ref
        for op in self.ops:
            if op["op"] in ("equal", "substitute"):
                for k, i in enumerate(range(op["ref_start"], op["ref_end"])):
                    out[i] = op["hyp_start"] + k
        return out

    def last_aligned_ref_index(self) -> int | None:
        """Index of the last reference word aligned to a hypothesis word."""
        last = None
        for op in self.ops:
            if op["op"] in ("equal", "substitute"):
                last = op["ref_end"] - 1
        return last

    def longest_deletion_run(self) -> int:
        """Longest run of consecutive reference words that were deleted.

        Adjacent ``delete`` chunks separated only by ``insert`` chunks (which
        consume no reference word) count as one run.
        """
        best = 0
        run = 0
        prev_end = None
        for op in self.ops:
            if op["op"] == "delete":
                length = op["ref_end"] - op["ref_start"]
                if prev_end is not None and op["ref_start"] == prev_end:
                    run += length
                else:
                    run = length
                prev_end = op["ref_end"]
                best = max(best, run)
            elif op["op"] == "insert":
                continue  # does not touch the reference, does not break the run
            else:
                run = 0
                prev_end = None
        return best

    def counts(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "n_ref": self.n_ref,
            "n_hyp": self.n_hyp,
        }


def _ops_from_jiwer(chunks: Sequence) -> list[dict]:
    return [
        {
            "op": c.type,
            "ref_start": int(c.ref_start_idx),
            "ref_end": int(c.ref_end_idx),
            "hyp_start": int(c.hyp_start_idx),
            "hyp_end": int(c.hyp_end_idx),
        }
        for c in chunks
    ]


def align_words(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    """Global monotonic alignment of two already-normalized word lists.

    Both degenerate cases are handled explicitly because ``jiwer`` refuses empty
    inputs: an empty hypothesis is all deletions, an empty reference is all
    insertions.
    """
    ref = list(reference)
    hyp = list(hypothesis)

    if not ref and not hyp:
        return Alignment(ref, hyp, [], 0, 0, 0, 0)
    if not hyp:
        ops = [{"op": "delete", "ref_start": 0, "ref_end": len(ref), "hyp_start": 0, "hyp_end": 0}]
        return Alignment(ref, hyp, ops, 0, 0, len(ref), 0)
    if not ref:
        ops = [{"op": "insert", "ref_start": 0, "ref_end": 0, "hyp_start": 0, "hyp_end": len(hyp)}]
        return Alignment(ref, hyp, ops, 0, 0, 0, len(hyp))

    import jiwer  # local import so the module can be imported without jiwer

    # The words are already normalized (no whitespace inside a token), so the
    # only transform we want is a plain whitespace split - jiwer's default
    # transform would additionally strip/collapse and is not identity-safe.
    transform = jiwer.Compose([jiwer.ReduceToListOfListOfWords(word_delimiter=" ")])
    out = jiwer.process_words(
        " ".join(ref),
        " ".join(hyp),
        reference_transform=transform,
        hypothesis_transform=transform,
    )
    ops = _ops_from_jiwer(out.alignments[0])
    return Alignment(
        reference=ref,
        hypothesis=hyp,
        ops=ops,
        hits=int(out.hits),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
    )

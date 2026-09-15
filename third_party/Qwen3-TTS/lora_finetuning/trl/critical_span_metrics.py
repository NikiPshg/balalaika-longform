"""Dependency-light typed critical-span alignment metrics for Russian TTS.

The training reward and held-out evaluator share the span contract, but not
their model/scorer state.  A span is anchored to normalized reference token
indices so repeated names or digit words cannot silently select the first
matching occurrence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import unicodedata
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0
    correct: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.reference_length:
            return self.errors / self.reference_length
        return 0.0 if self.errors == 0 else 1.0


@dataclass(frozen=True)
class CriticalSpanScore:
    span_type: str
    token_start: int
    token_end: int
    gold_tokens: tuple[str, ...]
    hypothesis_tokens: tuple[str, ...]
    word_counts: EditCounts
    char_counts: EditCounts

    @property
    def wer(self) -> float:
        return self.word_counts.rate

    @property
    def cer(self) -> float:
        return self.char_counts.rate

    @property
    def exact(self) -> bool:
        return self.word_counts.errors == 0 and self.char_counts.errors == 0

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update({"wer": self.wer, "cer": self.cer, "exact": self.exact})
        return value


@dataclass(frozen=True)
class TypedCriticalResult:
    spans: tuple[CriticalSpanScore, ...]
    utterance_word_counts: EditCounts
    utterance_char_counts: EditCounts

    @property
    def utterance_wer(self) -> float:
        return self.utterance_word_counts.rate

    @property
    def utterance_cer(self) -> float:
        return self.utterance_char_counts.rate

    @property
    def critical_wer(self) -> float:
        return _merged_counts(self.spans, "word_counts").rate

    @property
    def critical_cer(self) -> float:
        return _merged_counts(self.spans, "char_counts").rate

    @property
    def critical_exact_rate(self) -> float:
        return sum(span.exact for span in self.spans) / len(self.spans)

    def as_dict(self) -> dict[str, Any]:
        return {
            "spans": [span.as_dict() for span in self.spans],
            "utterance_word_counts": asdict(self.utterance_word_counts),
            "utterance_char_counts": asdict(self.utterance_char_counts),
            "utterance_wer": self.utterance_wer,
            "utterance_cer": self.utterance_cer,
            "critical_wer": self.critical_wer,
            "critical_cer": self.critical_cer,
            "critical_exact_rate": self.critical_exact_rate,
        }


@dataclass(frozen=True)
class _Operation:
    tag: str
    reference_index: int | None
    hypothesis_index: int | None


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value).lower().replace("ё", "е").replace("+", ""))
    characters: list[str] = []
    for character in text:
        if unicodedata.category(character) == "Mn":
            continue
        characters.append(character if character.isalnum() else " ")
    return " ".join("".join(characters).split())


def _operations(reference: Sequence[str], hypothesis: Sequence[str]) -> list[_Operation]:
    rows, columns = len(reference), len(hypothesis)
    distances = [[0] * (columns + 1) for _ in range(rows + 1)]
    for index in range(rows + 1):
        distances[index][0] = index
    for index in range(columns + 1):
        distances[0][index] = index
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            cost = reference[row - 1] != hypothesis[column - 1]
            distances[row][column] = min(
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
                distances[row - 1][column - 1] + cost,
            )
    row, column = rows, columns
    reversed_operations: list[_Operation] = []
    while row or column:
        if row and column:
            cost = reference[row - 1] != hypothesis[column - 1]
            if distances[row][column] == distances[row - 1][column - 1] + cost:
                reversed_operations.append(
                    _Operation("sub" if cost else "eq", row - 1, column - 1)
                )
                row -= 1
                column -= 1
                continue
        if row and distances[row][column] == distances[row - 1][column] + 1:
            reversed_operations.append(_Operation("del", row - 1, None))
            row -= 1
        else:
            reversed_operations.append(_Operation("ins", None, column - 1))
            column -= 1
    return list(reversed(reversed_operations))


def _counts(operations: Sequence[_Operation], reference_length: int) -> EditCounts:
    return EditCounts(
        substitutions=sum(item.tag == "sub" for item in operations),
        deletions=sum(item.tag == "del" for item in operations),
        insertions=sum(item.tag == "ins" for item in operations),
        reference_length=reference_length,
        correct=sum(item.tag == "eq" for item in operations),
    )


def _merged_counts(spans: Sequence[CriticalSpanScore], field: str) -> EditCounts:
    values = [getattr(span, field) for span in spans]
    return EditCounts(
        substitutions=sum(value.substitutions for value in values),
        deletions=sum(value.deletions for value in values),
        insertions=sum(value.insertions for value in values),
        reference_length=sum(value.reference_length for value in values),
        correct=sum(value.correct for value in values),
    )


def _spoken_tokens(span: Mapping[str, Any]) -> tuple[str, ...]:
    value = span.get("spoken_gold")
    if isinstance(value, str):
        tokens = normalize_text(value).split()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        tokens = [token for part in value for token in normalize_text(part).split()]
    else:
        tokens = []
    if not tokens:
        raise ValueError("critical span spoken_gold must contain normalized tokens")
    return tuple(tokens)


def _resolve_span(
    reference_tokens: Sequence[str], span: Mapping[str, Any]
) -> tuple[str, int, int, tuple[str, ...]]:
    span_type = str(span.get("type") or "").strip()
    if not span_type:
        raise ValueError("critical span type must not be empty")
    gold = _spoken_tokens(span)
    start, end = span.get("token_start"), span.get("token_end")
    if start is not None or end is not None:
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(reference_tokens)
        ):
            raise ValueError("critical span token offsets are invalid")
        if tuple(reference_tokens[start:end]) != gold:
            raise ValueError("critical span offsets do not match spoken_gold")
        return span_type, start, end, gold
    matches = [
        index
        for index in range(len(reference_tokens) - len(gold) + 1)
        if tuple(reference_tokens[index : index + len(gold)]) == gold
    ]
    if not matches:
        raise ValueError("critical span is absent from normalized reference")
    if len(matches) != 1:
        raise ValueError("critical span is ambiguous; explicit token offsets are required")
    return span_type, matches[0], matches[0] + len(gold), gold


def _span_operations(
    operations: Sequence[_Operation],
    hypothesis_tokens: Sequence[str],
    start: int,
    end: int,
) -> tuple[list[_Operation], tuple[str, ...]]:
    selected: list[_Operation] = []
    hypothesis: list[str] = []
    consumed_reference = 0
    for operation in operations:
        in_span = (
            start <= consumed_reference <= end
            if operation.tag == "ins"
            else start <= int(operation.reference_index) < end
        )
        if in_span:
            selected.append(operation)
            if operation.hypothesis_index is not None:
                hypothesis.append(hypothesis_tokens[operation.hypothesis_index])
        if operation.reference_index is not None:
            consumed_reference += 1
    return selected, tuple(hypothesis)


def infer_changed_token_span(input_text: str, normalized_gold: str) -> tuple[int, int, tuple[str, ...]]:
    """Infer the contiguous gold-token region replacing a raw written span."""

    input_tokens = normalize_text(input_text).split()
    gold_tokens = normalize_text(normalized_gold).split()
    changed = [
        operation.hypothesis_index
        for operation in _operations(input_tokens, gold_tokens)
        if operation.tag in {"sub", "ins"} and operation.hypothesis_index is not None
    ]
    if not changed:
        raise ValueError("input and normalized gold have no identifiable changed region")
    start, end = min(changed), max(changed) + 1
    return start, end, tuple(gold_tokens[start:end])


def score_typed_text_pair(
    reference: str,
    hypothesis: str,
    critical_spans: Sequence[Mapping[str, Any]],
) -> TypedCriticalResult:
    """Score an utterance and every explicitly typed critical span."""

    if not isinstance(critical_spans, Sequence) or isinstance(critical_spans, (str, bytes)):
        raise TypeError("critical_spans must be a sequence of mappings")
    if not critical_spans:
        raise ValueError("at least one critical span is required")
    reference_tokens = normalize_text(reference).split()
    hypothesis_tokens = normalize_text(hypothesis).split()
    operations = _operations(reference_tokens, hypothesis_tokens)
    utterance_word_counts = _counts(operations, len(reference_tokens))
    reference_characters = list("".join(reference_tokens))
    hypothesis_characters = list("".join(hypothesis_tokens))
    utterance_char_counts = _counts(
        _operations(reference_characters, hypothesis_characters), len(reference_characters)
    )
    scores: list[CriticalSpanScore] = []
    for raw_span in critical_spans:
        if not isinstance(raw_span, Mapping):
            raise TypeError("each critical span must be a mapping")
        span_type, start, end, gold = _resolve_span(reference_tokens, raw_span)
        span_ops, span_hypothesis = _span_operations(
            operations, hypothesis_tokens, start, end
        )
        word_counts = _counts(span_ops, len(gold))
        gold_characters = list("".join(gold))
        hypothesis_characters = list("".join(span_hypothesis))
        char_counts = _counts(
            _operations(gold_characters, hypothesis_characters), len(gold_characters)
        )
        scores.append(
            CriticalSpanScore(
                span_type=span_type,
                token_start=start,
                token_end=end,
                gold_tokens=gold,
                hypothesis_tokens=span_hypothesis,
                word_counts=word_counts,
                char_counts=char_counts,
            )
        )
    return TypedCriticalResult(tuple(scores), utterance_word_counts, utterance_char_counts)


def aggregate_typed_metrics(results: Sequence[TypedCriticalResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("typed aggregation requires at least one result")
    spans = [span for result in results for span in result.spans]
    by_type: dict[str, list[CriticalSpanScore]] = {}
    for span in spans:
        by_type.setdefault(span.span_type, []).append(span)

    def block(values: Sequence[CriticalSpanScore]) -> dict[str, Any]:
        words = _merged_counts(values, "word_counts")
        characters = _merged_counts(values, "char_counts")
        return {
            "span_count": len(values),
            "micro_wer": words.rate,
            "micro_cer": characters.rate,
            "macro_wer": sum(value.wer for value in values) / len(values),
            "macro_cer": sum(value.cer for value in values) / len(values),
            "exact_rate": sum(value.exact for value in values) / len(values),
            "word_errors": words.errors,
            "word_reference_length": words.reference_length,
            "char_errors": characters.errors,
            "char_reference_length": characters.reference_length,
        }

    utterance_words = EditCounts(
        substitutions=sum(x.utterance_word_counts.substitutions for x in results),
        deletions=sum(x.utterance_word_counts.deletions for x in results),
        insertions=sum(x.utterance_word_counts.insertions for x in results),
        reference_length=sum(x.utterance_word_counts.reference_length for x in results),
        correct=sum(x.utterance_word_counts.correct for x in results),
    )
    utterance_chars = EditCounts(
        substitutions=sum(x.utterance_char_counts.substitutions for x in results),
        deletions=sum(x.utterance_char_counts.deletions for x in results),
        insertions=sum(x.utterance_char_counts.insertions for x in results),
        reference_length=sum(x.utterance_char_counts.reference_length for x in results),
        correct=sum(x.utterance_char_counts.correct for x in results),
    )
    return {
        "row_count": len(results),
        "critical": block(spans),
        "per_type": {name: block(values) for name, values in sorted(by_type.items())},
        "micro_utterance_wer": utterance_words.rate,
        "micro_utterance_cer": utterance_chars.rate,
    }


def typed_critical_reward(
    result: TypedCriticalResult,
    *,
    wer_weight: float = 0.35,
    cer_weight: float = 0.45,
    exact_weight: float = 0.20,
    temperature: float = 3.0,
    worst_span_weight: float = 0.30,
) -> float:
    """Dense mean-plus-worst typed-span reward in the closed interval [0, 1]."""

    weights = (wer_weight, cer_weight, exact_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("typed reward weights must be finite/non-negative with positive sum")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("typed reward temperature must be finite and positive")
    if not math.isfinite(worst_span_weight) or not 0 <= worst_span_weight <= 1:
        raise ValueError("worst_span_weight must be in [0, 1]")
    rewards = []
    total = sum(weights)
    for span in result.spans:
        error = (
            wer_weight * min(1.0, span.wer)
            + cer_weight * min(1.0, span.cer)
            + exact_weight * float(not span.exact)
        ) / total
        rewards.append(1.0 - math.tanh(temperature * error))
    mean = sum(rewards) / len(rewards)
    return (1.0 - worst_span_weight) * mean + worst_span_weight * min(rewards)


__all__ = [
    "CriticalSpanScore",
    "EditCounts",
    "TypedCriticalResult",
    "aggregate_typed_metrics",
    "infer_changed_token_span",
    "normalize_text",
    "score_typed_text_pair",
    "typed_critical_reward",
]

"""Frozen text normalization for RuLongTTS evaluation (PLAN.md §9.4).

Two variants, both applied identically to reference and hypothesis:

    strict   NFKC -> lowercase -> ё→е -> non-kept chars to space -> collapse ws
    lenient  strict + digit runs expanded to Russian cardinal words

The contract lives in ``configs/text_normalization.yaml``; this module is its
implementation.  ``load_spec()`` reads the YAML and verifies that the values
this module actually implements are the values the YAML promises, so the two
cannot silently drift apart.

The number-to-words routine is a deterministic pure-python implementation on
purpose: it must not change when a pip package is upgraded.  It is cross-checked
against ``num2words(lang='ru')`` in tests/test_eval_known_cases.py.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "NORMALIZATION_VERSION",
    "normalize",
    "normalize_strict",
    "normalize_lenient",
    "words",
    "number_to_russian_words",
    "load_spec",
    "spec_path",
]

NORMALIZATION_VERSION = 1

_CYRILLIC = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
_LATIN = "abcdefghijklmnopqrstuvwxyz"
_DIGITS = "0123456789"
# ё is intentionally absent: it is mapped to е before filtering.
_KEEP = set(_CYRILLIC + _LATIN + _DIGITS)

_WS_RE = re.compile(r"\s+")
_DIGIT_RUN_RE = re.compile(r"\d+")

MAX_EXPANDABLE_DIGITS = 24

# ---------------------------------------------------------------------------
# Russian cardinal numbers (masculine, nominative)
# ---------------------------------------------------------------------------

_UNITS_M = [
    "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
]
_UNITS_F = [
    "ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
]
_TEENS = [
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
]
_TENS = [
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят",
    "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
]
_HUNDREDS = [
    "", "сто", "двести", "триста", "четыреста", "пятьсот",
    "шестьсот", "семьсот", "восемьсот", "девятьсот",
]

# (singular, paucal 2-4, plural) + grammatical gender of the unit digits
_SCALES: list[tuple[str, str, str, str]] = [
    ("", "", "", "m"),
    ("тысяча", "тысячи", "тысяч", "f"),
    ("миллион", "миллиона", "миллионов", "m"),
    ("миллиард", "миллиарда", "миллиардов", "m"),
    ("триллион", "триллиона", "триллионов", "m"),
    ("квадриллион", "квадриллиона", "квадриллионов", "m"),
    ("квинтиллион", "квинтиллиона", "квинтиллионов", "m"),
    ("секстиллион", "секстиллиона", "секстиллионов", "m"),
    ("септиллион", "септиллиона", "септиллионов", "m"),
]


def _plural_form(n: int, forms: tuple[str, str, str]) -> str:
    """Russian numeric agreement: 1 -> forms[0], 2-4 -> forms[1], else forms[2]."""
    n = abs(n) % 100
    if 11 <= n <= 19:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def _triplet_words(value: int, gender: str) -> list[str]:
    """Words for a value in 1..999."""
    out: list[str] = []
    hundreds, rest = divmod(value, 100)
    if hundreds:
        out.append(_HUNDREDS[hundreds])
    tens, units = divmod(rest, 10)
    if tens == 1:
        out.append(_TEENS[units])
    else:
        if tens:
            out.append(_TENS[tens])
        if units:
            out.append(_UNITS_F[units] if gender == "f" else _UNITS_M[units])
    return out


def number_to_russian_words(value: int) -> str:
    """Deterministic Russian cardinal (masculine, nominative) for a non-negative int.

    >>> number_to_russian_words(0)
    'ноль'
    >>> number_to_russian_words(1990)
    'одна тысяча девятьсот девяносто'
    """
    if value < 0:
        return "минус " + number_to_russian_words(-value)
    if value == 0:
        return "ноль"

    triplets: list[int] = []
    rest = value
    while rest:
        rest, tri = divmod(rest, 1000)
        triplets.append(tri)

    if len(triplets) > len(_SCALES):
        # Beyond septillions: read digit by digit (deterministic fallback).
        return " ".join(_UNITS_M[int(ch)] for ch in str(value))

    out: list[str] = []
    for idx in range(len(triplets) - 1, -1, -1):
        tri = triplets[idx]
        if tri == 0:
            continue
        singular, paucal, plural, gender = _SCALES[idx]
        out.extend(_triplet_words(tri, gender))
        if singular:
            out.append(_plural_form(tri, (singular, paucal, plural)))
    return " ".join(out)


def _expand_digit_run(match: "re.Match[str]") -> str:
    run = match.group(0)
    if len(run) > MAX_EXPANDABLE_DIGITS:
        return " " + " ".join(_UNITS_M[int(ch)] for ch in run) + " "
    stripped = run.lstrip("0")
    if stripped != run and stripped:
        # Leading zeros ("007"): read digit by digit, that is how they are spoken.
        return " " + " ".join(_UNITS_M[int(ch)] for ch in run) + " "
    return " " + number_to_russian_words(int(run)) + " "


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _base_normalize(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = text.replace("ё", "е").replace("Ё", "е")
    text = "".join(ch if ch in _KEEP else " " for ch in text)
    return _WS_RE.sub(" ", text).strip()


def normalize_strict(text: str) -> str:
    """Frozen strict normalization (digits kept literal)."""
    return _base_normalize(text)


def normalize_lenient(text: str) -> str:
    """Frozen lenient normalization (digit runs expanded to Russian words)."""
    base = _base_normalize(text)
    if not base or not any(ch.isdigit() for ch in base):
        return base
    expanded = _DIGIT_RUN_RE.sub(_expand_digit_run, base)
    return _WS_RE.sub(" ", expanded).strip()


def normalize(text: str, variant: str = "lenient") -> str:
    """Normalize ``text`` with ``variant`` in {"strict", "lenient"}."""
    if variant == "strict":
        return normalize_strict(text)
    if variant == "lenient":
        return normalize_lenient(text)
    raise ValueError(f"unknown normalization variant: {variant!r}")


def words(text: str, variant: str = "lenient") -> list[str]:
    """Normalized whitespace-separated word list."""
    normalized = normalize(text, variant)
    return normalized.split() if normalized else []


def normalize_many(texts: Iterable[str], variant: str = "lenient") -> list[str]:
    return [normalize(t, variant) for t in texts]


# ---------------------------------------------------------------------------
# Spec file
# ---------------------------------------------------------------------------


def spec_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "text_normalization.yaml"


def load_spec(path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/text_normalization.yaml and assert it matches this module.

    Raises RuntimeError when the YAML asks for behaviour this module does not
    implement (so a config edit cannot silently do nothing).
    """
    import yaml  # local import: keeps the module importable without pyyaml

    p = Path(path) if path is not None else spec_path()
    with open(p, "rt", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    common = spec.get("common", {})
    problems: list[str] = []
    if common.get("unicode_form") != "NFKC":
        problems.append("common.unicode_form must be NFKC")
    if common.get("lowercase") is not True:
        problems.append("common.lowercase must be true")
    if common.get("yo_to_e") is not True:
        problems.append("common.yo_to_e must be true")
    if common.get("non_keep_char_action") != "replace_with_space":
        problems.append("common.non_keep_char_action must be replace_with_space")
    if common.get("collapse_whitespace") is not True:
        problems.append("common.collapse_whitespace must be true")
    variants = spec.get("variants", {})
    if variants.get("strict", {}).get("expand_digits") is not False:
        problems.append("variants.strict.expand_digits must be false")
    if variants.get("lenient", {}).get("expand_digits") is not True:
        problems.append("variants.lenient.expand_digits must be true")
    if spec.get("primary_variant") not in ("strict", "lenient"):
        problems.append("primary_variant must be strict or lenient")
    if problems:
        raise RuntimeError(
            "configs/text_normalization.yaml disagrees with src/eval/normalize.py: "
            + "; ".join(problems)
        )
    return spec

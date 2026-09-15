"""A2's reference implementation of the frozen evaluation normalizer.

PLAN.md §14 lists ``src/eval/normalize.py`` under BOTH A2 and A4.  A4 created it
first (2026-08-27, see reports/decisions.md), so A2 does **not** overwrite it.
This module is A2's independent implementation of the contract written down in
reports/normalization_spec.md and configs/text_normalization.yaml; running it as
a script proves that the two implementations agree character for character on
the frozen rules, and measures the one place where A2 proposes to change them.

    python src/eval/normalize_spec_ref.py            # equivalence + measurements

Frozen pipeline (identical for reference and hypothesis, PLAN.md §9.4):

    NFKC -> lowercase -> ё→е -> every char outside [а-я a-z 0-9] becomes a
    single space -> collapse whitespace -> strip
      strict :  digit runs are kept as literal tokens
      lenient:  digit runs are expanded to Russian cardinal words

A2 decisions carried by this module (see reports/normalization_spec.md):
  * hyphen  -> space  ("какой-то" == "какой то"), applied to both sides;
  * apostrophe -> space (no Russian word needs one; benchmark texts have none);
  * Latin letters are KEPT (a Latin insertion must stay visible as an error),
    but benchmark ``text_tts`` may not contain any;
  * PROPOSED v2: the leading "один/одна" before a scale word is dropped
    ("1990" -> "тысяча девятьсот девяносто"), because the corpus says so.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC_REF_VERSION = 1

_CYR = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
_LAT = "abcdefghijklmnopqrstuvwxyz"
_DIG = "0123456789"
_KEEP = frozenset(_CYR + _LAT + _DIG)
_WS = re.compile(r"\s+")
_RUN = re.compile(r"\d+")
MAX_EXPANDABLE_DIGITS = 24

_U_M = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_U_F = ["ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEEN = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
         "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят",
         "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_HUND = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
         "шестьсот", "семьсот", "восемьсот", "девятьсот"]
_SCALES = [("", "", "", "m"),
           ("тысяча", "тысячи", "тысяч", "f"),
           ("миллион", "миллиона", "миллионов", "m"),
           ("миллиард", "миллиарда", "миллиардов", "m"),
           ("триллион", "триллиона", "триллионов", "m"),
           ("квадриллион", "квадриллиона", "квадриллионов", "m"),
           ("квинтиллион", "квинтиллиона", "квинтиллионов", "m"),
           ("секстиллион", "секстиллиона", "секстиллионов", "m"),
           ("септиллион", "септиллиона", "септиллионов", "m")]


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    n = abs(n) % 100
    if 11 <= n <= 19:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def _triplet(value: int, gender: str) -> list[str]:
    out: list[str] = []
    h, rest = divmod(value, 100)
    if h:
        out.append(_HUND[h])
    t, u = divmod(rest, 10)
    if t == 1:
        out.append(_TEEN[u])
    else:
        if t:
            out.append(_TENS[t])
        if u:
            out.append(_U_F[u] if gender == "f" else _U_M[u])
    return out


def number_to_russian_words_ref(value: int, drop_leading_one: bool = False) -> str:
    """Russian cardinal, masculine nominative.

    ``drop_leading_one`` implements A2's PROPOSED v2 rule: a most-significant
    triplet equal to 1 loses its "один"/"одна" before the scale word, so
    1990 reads "тысяча девятьсот девяносто" (the spoken and corpus form)
    rather than "одна тысяча девятьсот девяносто".
    """
    if value < 0:
        return "минус " + number_to_russian_words_ref(-value, drop_leading_one)
    if value == 0:
        return "ноль"
    triplets: list[int] = []
    rest = value
    while rest:
        rest, tri = divmod(rest, 1000)
        triplets.append(tri)
    if len(triplets) > len(_SCALES):
        return " ".join(_U_M[int(c)] for c in str(value))
    out: list[str] = []
    top = len(triplets) - 1
    for idx in range(top, -1, -1):
        tri = triplets[idx]
        if tri == 0:
            continue
        singular, paucal, plural, gender = _SCALES[idx]
        words = _triplet(tri, gender)
        if drop_leading_one and idx == top and idx > 0 and tri == 1:
            words = []          # "тысяча ...", not "одна тысяча ..."
        out.extend(words)
        if singular:
            out.append(_plural(tri, (singular, paucal, plural)))
    return " ".join(out)


def _base(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().replace("ё", "е")
    text = "".join(c if c in _KEEP else " " for c in text)
    return _WS.sub(" ", text).strip()


def normalize_ref(text: str, variant: str = "lenient", drop_leading_one: bool = False) -> str:
    base = _base(text)
    if variant == "strict":
        return base
    if variant != "lenient":
        raise ValueError(f"unknown variant {variant!r}")
    if not base or not any(c.isdigit() for c in base):
        return base

    def repl(m: "re.Match[str]") -> str:
        run = m.group(0)
        if len(run) > MAX_EXPANDABLE_DIGITS or (run.lstrip("0") != run and run.lstrip("0")):
            return " " + " ".join(_U_M[int(c)] for c in run) + " "
        return " " + number_to_russian_words_ref(int(run), drop_leading_one) + " "

    return _WS.sub(" ", _RUN.sub(repl, base)).strip()


# ---------------------------------------------------------------------------
# equivalence check against A4's src/eval/normalize.py
# ---------------------------------------------------------------------------


def sample_texts(limit: int = 4000) -> list[str]:
    """Deterministic corpus sample: every text field of the first rows of all.jsonl."""
    path = ROOT / "data" / "manifests" / "all.jsonl"
    out: list[str] = []
    if not path.exists():
        return out
    with open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            row = json.loads(line)
            out.append(row.get("text") or "")
            out.append(row.get("text_e2e") or "")
    return out


UNIT_CASES = [
    "Ёлка, ёж — «ЁЖИК»!",
    "какой-то по-моему из-за д'Артаньян",
    "в 1990 году, 007, 3.14 и 1 000 000",
    "ГОСТ 12.1.005-88, № 46/12",
    "OpenAI и Zoom",
    "",
    "   ",
    "2024-й, 90-х, 1-е место",
]


def main() -> int:
    from src.eval import normalize as a4

    texts = UNIT_CASES + sample_texts()
    n = 0
    mismatch_strict = 0
    mismatch_lenient = 0
    first: list[str] = []
    for t in texts:
        n += 1
        if a4.normalize_strict(t) != normalize_ref(t, "strict"):
            mismatch_strict += 1
            if len(first) < 5:
                first.append(f"STRICT {t[:60]!r}")
        if a4.normalize_lenient(t) != normalize_ref(t, "lenient"):
            mismatch_lenient += 1
            if len(first) < 5:
                first.append(f"LENIENT {t[:60]!r}")
    print(f"texts compared: {n}")
    print(f"strict  mismatches vs src/eval/normalize.py: {mismatch_strict}")
    print(f"lenient mismatches vs src/eval/normalize.py: {mismatch_lenient}")
    for f in first:
        print("  ", f)

    # number routine, 0..200000 plus decades of powers of ten
    values = list(range(0, 20001)) + [10**k for k in range(3, 13)] + \
             [1990, 2024, 1000000, 1000000000, 21000, 101000]
    diff = 0
    for v in values:
        if a4.number_to_russian_words(v) != number_to_russian_words_ref(v, False):
            diff += 1
    print(f"number_to_russian_words: {len(values)} values, {diff} mismatches (frozen rule)")

    v2 = sum(
        1 for v in values
        if number_to_russian_words_ref(v, True) != number_to_russian_words_ref(v, False)
    )
    print(f"PROPOSED v2 (drop leading one) changes {v2}/{len(values)} of those values")
    for v in (1990, 1000, 1_000_000, 21000, 2024):
        print(f"   {v}: frozen={number_to_russian_words_ref(v, False)!r} "
              f"v2={number_to_russian_words_ref(v, True)!r}")
    return 0 if (mismatch_strict == 0 and mismatch_lenient == 0 and diff == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())

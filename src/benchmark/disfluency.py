"""Frozen disfluency filter for RuLongTTS benchmark texts (PLAN.md §6.5).

Training text keeps every disfluency; benchmark text must be clean.  This module
implements the contract in ``configs/disfluency_filter.yaml``:

* hesitation sounds (э-э, м-м, а-а, эм, угу, ...) are always removed;
* discourse fillers (ну, вот, типа, как бы, короче, значит, ...) are removed
  unless a listed syntactic context makes them content words;
* an immediately repeated word loses its first occurrence;
* a truncated word ("прос-", "эконо экономика") is removed;
* a stuttered onset rendered by the ASR as a doubled first letter ("Ррусская")
  loses the duplicated letter.

Everything is expressed as rules over words, never as per-document exception
lists, because the parquet is still being filtered and the chosen roots will
change (decisions.md, 2026-08-27).

`scan()` returns the edits, `apply_edits()` performs them, `clean()` does both.
`density()` is the selection statistic (filler tokens per 100 word tokens).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "DISFLUENCY_VERSION",
    "Edit",
    "Token",
    "tokenize",
    "norm_token",
    "scan",
    "apply_edits",
    "clean",
    "clean_pieces",
    "finalize",
    "apply_charset",
    "ALLOWED_CHARS",
    "expand_edits",
    "density",
    "repair_casing",
    "lowercase_vocabulary",
    "load_spec",
    "spec_path",
]

DISFLUENCY_VERSION = 1

# --- frozen rule tables (mirrored by configs/disfluency_filter.yaml) --------

HESITATION_RE = re.compile(
    r"^(?:э+(?:-э+)*|м+(?:-м+)*|а+-а+(?:-а+)*|ы+|э-?эм|эм|гм|хм|мгм|угу|ага|мда)$"
)

DISCOURSE_FILLERS: dict[str, dict[str, frozenset[str]]] = {
    "ну": {"keep_if_next": frozenset(), "keep_if_prev": frozenset()},
    "вот": {
        "keep_if_next": frozenset(
            "почему зачем кто что как где когда такой такая такое такие".split()
        ),
        "keep_if_prev": frozenset(),
    },
    "типа": {"keep_if_next": frozenset(), "keep_if_prev": frozenset()},
    "короче": {"keep_if_next": frozenset(), "keep_if_prev": frozenset()},
    "значит": {
        "keep_if_next": frozenset({"что", "ли"}),
        "keep_if_prev": frozenset({"это", "что", "все", "то", "ничего", "значить"}),
    },
}

MULTIWORD_FILLERS: tuple[tuple[str, ...], ...] = (
    ("как", "бы"),
    ("короче", "говоря"),
    ("так", "сказать"),
    ("скажем", "так"),
)

REPEAT_KEEP = frozenset("очень самый самая самое чуть еле давно".split())

# A speech restart that a prefix rule cannot see because the short member is a
# frequent function word.  Added after the manual review of the pilot roots.
RESTART_BIGRAMS: tuple[tuple[str, str], ...] = (
    ("так", "такой"), ("так", "такая"), ("так", "такое"), ("так", "такие"),
)

# The "token is a prefix of the next token" truncation rule must not fire on
# ordinary function words ("по полочкам", "за закон", "как какой", "то только").
FUNCTION_WORDS = frozenset("""
в во на над под по за из изо от ото до к ко с со о об обо при про для без у
я ты он она оно мы вы они мне тебе ему ей нам вам им меня тебя его ее нас вас их
себя себе а и но да же ли бы не ни то так как что кто где куда когда откуда
зачем почему чем или либо если чтобы хотя пока ведь лишь даже уже еще вон
там тут тот та те это эта эти этот весь вся все сам сама само сами
мой моя мое твой наш ваш свой ну вот раз потом затем тоже также
очень более менее много мало нет есть был была были было будет быть
""".split())

PREFIX_MIN_LEN = 2
PREFIX_MIN_GAP = 2
STUTTER_MIN_LEN = 4
# Ordinary Russian words really do start with these doubled consonants
# ("ввод", "вверх", "ссылка", "ссора", "жжение"), so they are never stutters.
STUTTER_EXCLUDED_ONSETS = ("вв", "сс", "жж")
_VOWELS = set("аеиоуыэюяё")

_TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:[-’'][А-Яа-яЁёA-Za-z]+)*")
_TRAILING_HYPHEN_RE = re.compile(r"(?<![-\w])([А-Яа-яЁёA-Za-z]{1,})-(?=\s|$)")
_WS_RE = re.compile(r"\s+")
_SENT_END = ".!?…"


@dataclass(frozen=True)
class Token:
    text: str          # raw surface form
    start: int         # char offset in the source string
    end: int
    norm: str          # lowercase, ё->е


@dataclass(frozen=True)
class Edit:
    kind: str          # hesitation|discourse|multiword|repeat|truncated|stutter
    start: int         # char span to delete (token span; whitespace fixed later)
    end: int
    surface: str       # deleted surface form
    replacement: str   # "" for a deletion, a shorter word for stutter repair
    context: str       # +-40 chars of the original text, for the report

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def norm_token(s: str) -> str:
    return s.lower().replace("ё", "е").replace("’", "'")


def tokenize(text: str) -> list[Token]:
    return [
        Token(m.group(0), m.start(), m.end(), norm_token(m.group(0)))
        for m in _TOKEN_RE.finditer(text)
    ]


def _context(text: str, start: int, end: int, width: int = 40) -> str:
    left = text[max(0, start - width) : start]
    right = text[end : end + width]
    return _WS_RE.sub(" ", f"{left}[[{text[start:end]}]]{right}").strip()


def scan(text: str) -> list[Edit]:
    """Return the frozen list of disfluency edits for ``text`` (offset-sorted)."""
    toks = tokenize(text)
    n = len(toks)
    edits: list[Edit] = []
    consumed: set[int] = set()

    # 1. multiword fillers (checked first so "как бы" wins over "как")
    i = 0
    while i < n:
        for phrase in MULTIWORD_FILLERS:
            k = len(phrase)
            if i + k <= n and all(toks[i + j].norm == phrase[j] for j in range(k)):
                s, e = toks[i].start, toks[i + k - 1].end
                edits.append(Edit("multiword", s, e, text[s:e], "", _context(text, s, e)))
                consumed.update(range(i, i + k))
                i += k - 1
                break
        i += 1

    # 2. per-token rules
    for idx, t in enumerate(toks):
        if idx in consumed:
            continue
        prev_norm = toks[idx - 1].norm if idx > 0 else ""
        next_norm = toks[idx + 1].norm if idx + 1 < n else ""

        if HESITATION_RE.match(t.norm):
            edits.append(
                Edit("hesitation", t.start, t.end, t.text, "", _context(text, t.start, t.end))
            )
            consumed.add(idx)
            continue

        rule = DISCOURSE_FILLERS.get(t.norm)
        if rule is not None:
            if next_norm in rule["keep_if_next"] or prev_norm in rule["keep_if_prev"]:
                pass
            else:
                edits.append(
                    Edit("discourse", t.start, t.end, t.text, "", _context(text, t.start, t.end))
                )
                consumed.add(idx)
                continue

        # speech restart that the prefix rule cannot see ("вот так такая логика")
        if (t.norm, next_norm) in RESTART_BIGRAMS:
            edits.append(
                Edit("restart", t.start, t.end, t.text, "", _context(text, t.start, t.end))
            )
            consumed.add(idx)
            continue

        # truncated word: proper prefix of the following token
        if (
            idx + 1 < n
            and len(t.norm) >= PREFIX_MIN_LEN
            and t.norm not in FUNCTION_WORDS
            and toks[idx + 1].norm.startswith(t.norm)
            and len(toks[idx + 1].norm) - len(t.norm) >= PREFIX_MIN_GAP
        ):
            edits.append(
                Edit("truncated", t.start, t.end, t.text, "", _context(text, t.start, t.end))
            )
            consumed.add(idx)
            continue

        # immediate repeat: drop the first of the pair
        if (
            idx + 1 < n
            and t.norm == toks[idx + 1].norm
            and t.norm not in REPEAT_KEEP
            and idx + 1 not in consumed
        ):
            edits.append(
                Edit("repeat", t.start, t.end, t.text, "", _context(text, t.start, t.end))
            )
            consumed.add(idx)
            continue

        # stuttered onset merged into one token by the ASR: "Ррусская"
        low = t.norm
        if (
            len(low) >= STUTTER_MIN_LEN
            and not low.startswith(STUTTER_EXCLUDED_ONSETS)
            and low[0] == low[1]
            and low[0] not in _VOWELS
            and low[0].isalpha()
            and low[2] in _VOWELS | set("рлмнвй")
        ):
            repaired = t.text[0] + t.text[2:]
            edits.append(
                Edit("stutter", t.start, t.end, t.text, repaired, _context(text, t.start, t.end))
            )
            consumed.add(idx)
            continue

    # 3. truncated words written with a trailing hyphen ("прос- простите")
    for m in _TRAILING_HYPHEN_RE.finditer(text):
        s, e = m.start(), m.end()
        if any(ed.start <= s < ed.end for ed in edits):
            continue
        edits.append(Edit("truncated", s, e, text[s:e], "", _context(text, s, e)))

    edits.sort(key=lambda ed: (ed.start, ed.end))
    # drop overlaps (keep the earliest / longest)
    out: list[Edit] = []
    last_end = -1
    for ed in edits:
        if ed.start < last_end:
            continue
        out.append(ed)
        last_end = ed.end
    return out


_CASE_SENT_END_RE = re.compile(r"[.!?…]\s*$")


def repair_casing(text: str, lower_seen: set[str] | None = None) -> tuple[str, int]:
    """Lowercase mid-sentence capitals that the ASR invented.

    gigaam-v3-e2e-ctc capitalizes words at random inside a sentence
    ("Так, рак вы меня Видите?").  A mid-sentence capitalized token is lowercased
    **only** when the same token also occurs in lowercase in the document, which
    leaves real proper nouns ("Пушкин", "Заозёрье") untouched.  Pass
    ``lower_seen`` to use document-level evidence while editing one piece.
    Returns (text, n_changed).
    """
    toks = tokenize(text)
    if lower_seen is None:
        lower_seen = {t.norm for t in toks if t.text[:1].islower()}
    out = list(text)
    changed = 0
    for t in toks:
        if not t.text[:1].isupper():
            continue
        left = text[: t.start]
        if not left.strip() or _CASE_SENT_END_RE.search(left):
            continue  # sentence-initial: keep the capital
        if t.norm in lower_seen:
            out[t.start] = t.text[0].lower()
            changed += 1
    return "".join(out), changed


def lowercase_vocabulary(text: str) -> set[str]:
    """Tokens that occur in lowercase somewhere in ``text``."""
    return {t.norm for t in tokenize(text) if t.text[:1].islower()}


_LETTER_RE = re.compile(r"[А-Яа-яЁёA-Za-z]")


def _sentence_has_other_letters(text: str, upto: int) -> bool:
    """True if the sentence containing ``upto`` already has letters before it."""
    cut = 0
    for i in range(upto - 1, -1, -1):
        if text[i] in _SENT_END:
            cut = i + 1
            break
    return bool(_LETTER_RE.search(text[cut:upto]))


def expand_edits(text: str, edits: Iterable[Edit]) -> list[Edit]:
    """Extend deletion spans over the punctuation they orphan.

    * a following comma is always swallowed  ("Ну, это" -> "это");
    * a following sentence terminator is swallowed only when the deleted token
      was the whole sentence ("Хорошо. Угу. Дальше." -> "Хорошо. Дальше.");
      otherwise the terminator is kept so the sentence still ends.
    """
    out: list[Edit] = []
    for ed in sorted(edits, key=lambda e: e.start):
        if ed.replacement:
            out.append(ed)
            continue
        end = ed.end
        j = end
        while j < len(text) and text[j] == " ":
            j += 1
        if j < len(text) and text[j] == ",":
            end = j + 1
        elif j < len(text) and text[j] in _SENT_END:
            if not _sentence_has_other_letters(text, ed.start):
                k = j
                while k < len(text) and text[k] in _SENT_END:
                    k += 1
                end = k
        out.append(Edit(ed.kind, ed.start, end, ed.surface, ed.replacement, ed.context))
    return out


def _splice(text: str, edits: Iterable[Edit]) -> str:
    pieces: list[str] = []
    cursor = 0
    for ed in sorted(edits, key=lambda e: e.start):
        if ed.start < cursor:
            continue
        pieces.append(text[cursor : ed.start])
        if ed.replacement:
            pieces.append(ed.replacement)
        cursor = ed.end
    pieces.append(text[cursor:])
    return "".join(pieces)


# --- frozen benchmark charset (configs/benchmark_pilot.yaml: text_charset) ---

QUOTE_CHARS = "«»\"\u201c\u201d\u201e\u201f\u2039\u203a\u201b\u275d\u275e"
APOSTROPHE_CHARS = "'\u2019\u2018`\u00b4\u02bc"
DASH_VARIANTS = ("---", "--", "\u2013", "\u2012", "\u2015", "\u2212", "\u2010", "\u2011")
EM_DASH = "\u2014"
ALLOWED_PUNCT = set(".,!?:;\u2026\u2014-")
ALLOWED_CHARS = ALLOWED_PUNCT | {" "}

_QUOTE_APOS_RE = re.compile(f"[{re.escape(QUOTE_CHARS + APOSTROPHE_CHARS)}]")
_INITIAL_DASH_RE = re.compile(r"(^|[.!?\u2026])(\s*)\u2014+\s*")
_GLUED_SENTENCE_RE = re.compile(r"([.!?\u2026])(?=[А-Яа-яЁёA-Za-z])")


def apply_charset(text: str) -> str:
    """Collapse the benchmark text onto one frozen character set.

    Every rule looks only to the LEFT of the character it changes, so the result
    for a prefix of ``text`` is the prefix of the result -- which is what keeps
    the nested B0..B4 chain of PLAN §7.2 intact.

    * every dash variant (including a ``--`` run) becomes U+2014;
    * an em dash in sentence-initial position is deleted (the ASR does not emit
      Russian dialogue dashes, so one there came from a ``--`` artifact);
    * every quotation mark is deleted (the ASR's quotes are unbalanced and are
      stripped by the frozen normalizer anyway, so text_ref is unchanged);
    * every apostrophe is deleted ("О'кей" -> "Окей");
    * a sentence terminator glued to the next word gets one space
      ("ингредиенты?.Могут" -> "ингредиенты?. Могут"), because the frontend is
      frozen to "" and would otherwise see one long token.
    """
    for variant in DASH_VARIANTS:
        text = text.replace(variant, EM_DASH)
    text = _QUOTE_APOS_RE.sub("", text)
    text = _INITIAL_DASH_RE.sub(lambda m: m.group(1) + (" " if m.group(1) else ""), text)
    text = _GLUED_SENTENCE_RE.sub(r"\1 ", text)
    return text


def _tidy(text: str) -> str:
    text = re.sub(r"\.{2,}", ".", text)          # ".." / "..." artifacts of the ASR
    text = re.sub(r",(\s*,)+", ",", text)
    text = re.sub(r"\s+([,.!?;:…])", r"\1", text)
    text = re.sub(r"([,;:])\s*([.!?…])", r"\2", text)
    text = re.sub(r"\s*—\s*—\s*", " — ", text)
    text = re.sub(r"^[\s,;:—.…!?-]+", "", text)
    return _WS_RE.sub(" ", text)


def finalize(text: str, max_passes: int = 6) -> str:
    """Deterministic punctuation/whitespace tidy + sentence-start capitalisation.

    The tidy rules feed each other (removing the space in "безмерна. . То" recreates
    the "…" run that the first rule collapses), so they are iterated to a fixed
    point; ``finalize`` is therefore idempotent, which is asserted in
    tests/test_benchmark.py.
    """
    text = apply_charset(text)
    text = _WS_RE.sub(" ", text)
    for _ in range(max_passes):
        nxt = _tidy(text)
        if nxt == text:
            break
        text = nxt
    return _recapitalize(text.strip())


def _recapitalize(text: str) -> str:
    """Capitalize the first letter of the string and of every sentence."""
    out = list(text)
    need_upper = True
    for i, ch in enumerate(out):
        if need_upper and ch.isalpha():
            out[i] = ch.upper()
            need_upper = False
        elif ch in _SENT_END:
            need_upper = True
        elif ch in "«\"'( ":
            continue
        elif ch.isalpha() or ch.isdigit():
            need_upper = False
    return "".join(out)


def apply_edits(text: str, edits: Iterable[Edit]) -> str:
    """Apply ``edits`` to ``text`` and tidy the punctuation deterministically."""
    return finalize(_splice(text, expand_edits(text, edits)))


def clean_pieces(pieces: list[str], max_passes: int = 5) -> tuple[list[str], list[Edit]]:
    """Clean a list of text pieces jointly, keeping the piece boundaries.

    The pieces are joined with a single space, scanned as one document (so that
    context rules and multi-token fillers see across a boundary), and the edits
    are then applied piece by piece.  ``finalize(" ".join(result[:k]))`` equals
    the corresponding prefix of ``finalize(" ".join(result))`` for every k that
    ends on a sentence boundary, which is what makes nested prefixes possible.
    """
    all_edits: list[Edit] = []
    current = list(pieces)
    for _ in range(max_passes):
        spans: list[tuple[int, int]] = []
        cursor = 0
        for i, p in enumerate(current):
            if i:
                cursor += 1  # the joining space
            spans.append((cursor, cursor + len(p)))
            cursor += len(p)
        text = " ".join(current)
        edits = expand_edits(text, scan(text))
        if not edits:
            break
        all_edits.extend(edits)
        out: list[str] = []
        for (s, e) in spans:
            buf: list[str] = []
            cur = s
            for ed in edits:
                if ed.end <= s or ed.start >= e:
                    continue
                a, b = max(ed.start, s), min(ed.end, e)
                if a > cur:
                    buf.append(text[cur:a])
                if ed.replacement and ed.start >= s and ed.end <= e:
                    buf.append(ed.replacement)
                cur = max(cur, b)
            buf.append(text[cur:e])
            out.append("".join(buf))
        current = out
    return current, all_edits


def clean(text: str) -> tuple[str, list[Edit]]:
    """Return (edited_text, edits)."""
    edits = scan(text)
    return apply_edits(text, edits), edits


def density(text: str) -> tuple[float, int, int]:
    """(filler tokens per 100 word tokens, n_filler_tokens, n_word_tokens)."""
    toks = tokenize(text)
    n_words = len(toks)
    n_fill = 0
    for ed in scan(text):
        n_fill += len(tokenize(ed.surface)) if ed.kind != "stutter" else 0
    return (100.0 * n_fill / n_words if n_words else 0.0), n_fill, n_words


# ---------------------------------------------------------------------------
# spec file
# ---------------------------------------------------------------------------


def spec_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "disfluency_filter.yaml"


def load_spec(path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/disfluency_filter.yaml and assert it matches this module."""
    import yaml

    p = Path(path) if path is not None else spec_path()
    with open(p, "rt", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    problems: list[str] = []
    if spec.get("hesitations_regex") != HESITATION_RE.pattern:
        problems.append("hesitations_regex differs from disfluency.HESITATION_RE")
    yaml_fillers = {k: v for k, v in (spec.get("discourse_fillers") or {}).items()
                    if v.get("enabled", True)}
    if set(yaml_fillers) != set(DISCOURSE_FILLERS):
        problems.append(
            f"discourse_fillers keys differ: yaml={sorted(yaml_fillers)} "
            f"code={sorted(DISCOURSE_FILLERS)}"
        )
    else:
        for k, v in yaml_fillers.items():
            if frozenset(v.get("keep_if_next") or []) != DISCOURSE_FILLERS[k]["keep_if_next"]:
                problems.append(f"discourse_fillers.{k}.keep_if_next differs")
            if frozenset(v.get("keep_if_prev") or []) != DISCOURSE_FILLERS[k]["keep_if_prev"]:
                problems.append(f"discourse_fillers.{k}.keep_if_prev differs")
    if tuple(tuple(x) for x in (spec.get("multiword_fillers") or [])) != MULTIWORD_FILLERS:
        problems.append("multiword_fillers differ")
    rep = spec.get("immediate_repeat") or {}
    if frozenset(rep.get("keep_pairs") or []) != REPEAT_KEEP:
        problems.append("immediate_repeat.keep_pairs differ")
    trunc = spec.get("truncated_word") or {}
    if trunc.get("prefix_of_next_min_len") != PREFIX_MIN_LEN:
        problems.append("truncated_word.prefix_of_next_min_len differs")
    if trunc.get("prefix_of_next_min_gap") != PREFIX_MIN_GAP:
        problems.append("truncated_word.prefix_of_next_min_gap differs")
    if (spec.get("stutter_initial") or {}).get("min_len") != STUTTER_MIN_LEN:
        problems.append("stutter_initial.min_len differs")
    if tuple((spec.get("stutter_initial") or {}).get("excluded_onsets") or []) != STUTTER_EXCLUDED_ONSETS:
        problems.append("stutter_initial.excluded_onsets differ")
    if frozenset(trunc.get("function_word_guard") or []) != FUNCTION_WORDS:
        problems.append("truncated_word.function_word_guard differs")
    if tuple(tuple(x) for x in (spec.get("restart_bigrams") or [])) != RESTART_BIGRAMS:
        problems.append("restart_bigrams differ")
    if problems:
        raise RuntimeError(
            "configs/disfluency_filter.yaml disagrees with src/benchmark/disfluency.py: "
            + "; ".join(problems)
        )
    return spec

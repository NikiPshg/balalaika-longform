"""PLAN §7.5 schema enforcement in scripts/run_generation.py (torch-free; runs in .venv-eval with pytest).

run_generation reads ONLY text_id/text_tts (benchmark) and voice_id/wav_path/ref_text (references);
a missing/empty canonical field is a hard SchemaError, never a fallback to a synonym.
"""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import run_generation as RG  # noqa: E402


def _jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return str(path)


def test_no_synonym_tables_left():
    src = open(os.path.join(ROOT, 'scripts', 'run_generation.py'), encoding='utf-8').read()
    for bad in ('TEXT_KEYS', 'TEXT_ID_KEYS', 'VOICE_ID_KEYS', 'REF_WAV_KEYS', 'REF_TEXT_KEYS', 'HUMAN_DUR_KEYS', 'def pick('):
        assert bad not in src, bad
    assert RG.BENCHMARK_FIELDS == ('text_id', 'text_tts')
    assert RG.REFERENCE_FIELDS == ('voice_id', 'wav_path', 'ref_text')


def test_benchmark_canonical_only(tmp_path):
    p = _jsonl(tmp_path / 'b.jsonl', [{'text_id': 'r__B0', 'text_tts': 'Привет.', 'text': 'IGNORED', 'human_duration_sec': 3.0},
                                       {'text_id': 'r__B1', 'text_tts': 'Ещё.'}])
    assert RG.load_benchmark(p) == [('r__B0', 'Привет.'), ('r__B1', 'Ещё.')]
    assert RG.load_benchmark(p, text_filter=r'__B[1]$') == [('r__B1', 'Ещё.')]


@pytest.mark.parametrize('rec', [
    {'text_id': 'x', 'text': 'synonym must not be accepted'},
    {'text_id': 'x', 'text_spoken': 'synonym must not be accepted'},
    {'id': 'x', 'text_tts': 'no text_id'},
    {'text_id': 'x', 'text_tts': ''},
    {'text_id': 'x', 'text_tts': None},
])
def test_benchmark_missing_field_is_hard_error(tmp_path, rec):
    p = _jsonl(tmp_path / 'b.jsonl', [rec])
    with pytest.raises(RG.SchemaError):
        RG.load_benchmark(p)


def test_benchmark_duplicate_text_id(tmp_path):
    p = _jsonl(tmp_path / 'b.jsonl', [{'text_id': 'x', 'text_tts': 'a'}, {'text_id': 'x', 'text_tts': 'b'}])
    with pytest.raises(RG.SchemaError):
        RG.load_benchmark(p)


def test_references_canonical_only(tmp_path):
    wav = tmp_path / 'v.wav'
    wav.write_bytes(b'RIFF')
    p = _jsonl(tmp_path / 'r.jsonl', [{'voice_id': 'v1', 'wav_path': 'v.wav', 'ref_text': 'т', 'audio_path': '/nonexistent', 'role': 'primary'},
                                       {'voice_id': 'v2', 'wav_path': str(wav), 'ref_text': 'т2'}])
    out = RG.load_references(p, root=str(tmp_path))
    assert out == [('v1', str(wav), 'т'), ('v2', str(wav), 'т2')]
    assert RG.load_references(p, voice_filter=r'v2$', root=str(tmp_path)) == [('v2', str(wav), 'т2')]


@pytest.mark.parametrize('rec', [
    {'voice_id': 'v', 'audio_path': 'v.wav', 'ref_text': 't'},   # synonym for wav_path
    {'voice_id': 'v', 'wav_path': 'v.wav', 'text': 't'},         # synonym for ref_text
    {'speaker_id': 'v', 'wav_path': 'v.wav', 'ref_text': 't'},   # synonym for voice_id
    {'voice_id': 'v', 'wav_path': 'missing.wav', 'ref_text': 't'},
])
def test_references_missing_field_is_hard_error(tmp_path, rec):
    (tmp_path / 'v.wav').write_bytes(b'RIFF')
    p = _jsonl(tmp_path / 'r.jsonl', [rec])
    with pytest.raises(RG.SchemaError):
        RG.load_references(p, root=str(tmp_path))


def test_real_project_files_parse():
    """The actual A2/A5 deliverables satisfy the canonical schema."""
    bench = os.path.join(ROOT, 'data', 'benchmark', 'pilot.jsonl')
    refs = os.path.join(ROOT, 'data', 'references', 'references.jsonl')
    if not (os.path.exists(bench) and os.path.exists(refs)):
        pytest.skip('pilot.jsonl / references.jsonl not built yet')
    texts = RG.load_benchmark(bench)
    assert len(texts) == 30 and len({t for t, _ in texts}) == 30
    # A2's root_ids are derived from the parquet and change on every rebuild, so
    # the filter is built from the file instead of being hard-coded.
    root = texts[0][0].rsplit('__', 1)[0]
    sel = RG.load_benchmark(bench, text_filter=re.escape(root) + r'__B[01]$')
    assert [t for t, _ in sel] == [root + '__B0', root + '__B1']
    voices = RG.load_references(refs, voice_filter=r'ref_(female|male)_01$')
    assert [v for v, _, _ in voices] == ['ref_female_01', 'ref_male_01']
    for _, wav, _ in voices:
        assert os.path.isabs(wav) and os.path.exists(wav)


def test_smoke_fixtures_if_built():
    bench = os.path.join(ROOT, 'data', 'smoke', 'benchmark.jsonl')
    refs = os.path.join(ROOT, 'data', 'smoke', 'references.jsonl')
    if not (os.path.exists(bench) and os.path.exists(refs)):
        pytest.skip('data/smoke not built (scripts/make_smoke_inputs.py)')
    assert [t for t, _ in RG.load_benchmark(bench)] == ['smoke_a_30s', 'smoke_b_2min', 'smoke_c_6min']
    (vid, wav, ref_text), = RG.load_references(refs)
    assert vid == 'smoke_v' and os.path.exists(wav) and ref_text

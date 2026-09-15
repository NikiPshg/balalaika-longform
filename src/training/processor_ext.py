"""Extra data-pipeline processors for the LLM-only SFT arms (A6). Used from configs/train/*.yaml via
`!name:src.training.processor_ext.<fn>` together with the stock cosyvoice.dataset.processor functions.

Why a separate module: the stock pipeline (filter -> resample -> compute_fbank -> parse_embedding ->
compute_whisper_fbank -> ... -> padding) decodes the audio, computes 80-mel + 128-mel whisper
features and a campplus embedding for every sample and extracts speech tokens ONLINE inside the
LLM forward (SpeechTokenExtractor). None of that is needed for CosyVoice3LM.forward when the
parquet rows already carry `speech_token` (llm.py: `if 'speech_token' not in batch: ...extract`),
and the stock `filter` silently DROPS samples with more than token_max_length text tokens /
max_length frames (PLAN §4.2 forbids silent filtering). This module therefore provides:

  duration_filter  E4 curriculum stage ceiling: the ONLY deliberate row filter here; it counts and
                 LOGS every dropped row and is disabled for the dev (cv) pass
  budget_check   explicit context-budget validation that RAISES on violation (no silent drop)
  sort_by_tokens buffer sort by target speech tokens (replacement for stock `sort`, which needs mel)
  token_batch    token-based dynamic batching (PLAN §11.3): close a batch when the sum of target
                 speech tokens or the padded positions (longest seq x n) would exceed the caps
  padding_llm    collate exactly the keys CosyVoice3LM.forward reads + per-batch token statistics

The repository is untouched; the stock functions still used are parquet_opener, tokenize, shuffle.
"""
import logging
import os
import sys

import torch
from torch.nn.utils.rnn import pad_sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.training.context_budget import CONTEXT_LIMIT, SAFETY_MARGIN, budget_row  # noqa: E402


class ContextBudgetError(RuntimeError):
    pass


class CurriculumFilterError(RuntimeError):
    pass


# The offline speech tokenizer runs at 25 Hz (configs/train/*.yaml `token_frame_rate`), so a row's
# duration can be recovered from its token count if the parquet `duration` column is ever missing.
TOKEN_FRAME_RATE = 25

# Per-process cumulative counters, used when the caller passes no `stats` dict. Keyed by pid because the
# dataloader workers are separate processes and each one runs its own copy of the pipeline.
_DURATION_FILTER_STATS = {}


def _row_duration_sec(sample):
    """Duration of a parquet row in seconds, or None if it cannot be determined.

    `duration` is written by src/training/make_parquet_arms.py for every row of both arms; the token-count
    fallback exists so that a hand-made parquet without that column is still measurable rather than silently
    unfiltered. NaN counts as unknown.
    """
    dur = sample.get('duration')
    if dur is not None:
        dur = float(dur)
        if dur == dur:  # not NaN
            return dur
    tok = sample.get('speech_token')
    if tok is not None:
        try:
            return len(tok) / float(TOKEN_FRAME_RATE)
        except TypeError:
            return None
    return None


def duration_filter(data, max_duration_sec=None, min_duration_sec=None, mode='train', stats=None, log_every=500):
    """E4 curriculum stage ceiling: pass a row only if min_duration_sec <= duration <= max_duration_sec.

    Pre-registered by the Lead in reports/decisions.md, 2026-08-29 ("ПРЕРЕГИСТРАЦИЯ E4 Curriculum-SFT"):
    three stages on the long arm with a growing ceiling on the duration of one training unit --
    S1 <= 90 s (500 steps), S2 <= 180 s (500 steps), S3 <= 900 s (2000 steps, i.e. the whole long arm, a
    no-op kept explicit). `None` on either side disables that side of the test.

    THIS IS THE ONE PROCESSOR IN THIS MODULE THAT DROPS ROWS ON PURPOSE. PLAN §4.2/§11.2 forbid SILENT
    filtering, not filtering: the curriculum ceiling is a deliberate, pre-registered part of the recipe, so
    every dropped row is counted and the counts are written to the training log --
      * once when the generator is created (= once per epoch per worker process), with the stage ceiling and
        the counts accumulated so far by this process,
      * every `log_every` rows while the epoch runs (0 disables the periodic line),
      * once more when the epoch's row stream ends, with that epoch's kept/dropped rows and hours.
    The same counters are accumulated into `stats` (a dict supplied by the caller, otherwise a per-pid entry
    of the module-level `_DURATION_FILTER_STATS`) so a test or a script can read them directly:
    keys `kept`, `dropped`, `kept_sec`, `dropped_sec`, `seen`, `epochs`, `active`, `mode`, `pid`,
    `max_duration_sec`, `min_duration_sec`.

    **In cv/dev mode NOTHING is filtered.** cosyvoice.dataset.dataset.Dataset builds the cv dataset with
    `mode='dev'` from the SAME `data_pipeline` list as the train dataset, so a ceiling applied there would
    make every stage compute its dev loss on a different subset of dev and the S1/S2/S3 losses would not be
    comparable with each other, with E2/E3, or with the checkpoint-selection rule (PLAN §11.5, dev-only).
    Filtering is therefore active only for `mode == 'train'`; in any other mode every row is passed through
    and the log line says so explicitly (`active=False`), so an unfiltered dev pass is never a silent
    assumption. Both facts -- the drop counts and the unfiltered dev -- are visible in the same train log.

    A row whose duration cannot be determined (no `duration` column, no `speech_token`, or NaN) RAISES
    `CurriculumFilterError` while filtering is active: a stage must never keep or drop a row whose length it
    cannot see. When filtering is inactive such a row is passed through and only counted (its seconds are
    not added to the hour totals).
    """
    if stats is None:
        stats = _DURATION_FILTER_STATS.setdefault(os.getpid(), {})
    for k in ('kept', 'dropped', 'seen', 'epochs'):
        stats.setdefault(k, 0)
    for k in ('kept_sec', 'dropped_sec'):
        stats.setdefault(k, 0.0)
    active = mode == 'train' and (max_duration_sec is not None or min_duration_sec is not None)
    pid = os.getpid()
    stats.update({'pid': pid, 'mode': mode, 'active': active,
                  'max_duration_sec': max_duration_sec, 'min_duration_sec': min_duration_sec})
    stats['epochs'] += 1
    epoch = stats['epochs']

    def _line(when, kept, dropped, kept_sec, dropped_sec):
        total = kept + dropped
        logging.info('duration_filter[pid %d] %s (epoch #%d, mode=%s, active=%s, ceiling max=%s s min=%s s): '
                     'kept %d/%d rows (%.2f h), dropped %d/%d rows (%.2f h, %.1f%%)',
                     pid, when, epoch, mode, active, max_duration_sec, min_duration_sec,
                     kept, total, kept_sec / 3600.0, dropped, total, dropped_sec / 3600.0,
                     100.0 * dropped / total if total else 0.0)

    _line('epoch start, cumulative for this process', stats['kept'], stats['dropped'],
          stats['kept_sec'], stats['dropped_sec'])
    kept = dropped = seen = 0
    kept_sec = dropped_sec = 0.0
    for sample in data:
        seen += 1
        stats['seen'] += 1
        dur = _row_duration_sec(sample)
        if dur is None:
            if active:
                raise CurriculumFilterError(
                    'duration unknown for utt {}: the curriculum stage (max_duration_sec={}, '
                    'min_duration_sec={}) cannot decide whether to keep it; the parquet row must carry '
                    '`duration` (or `speech_token`)'.format(sample.get('utt'), max_duration_sec, min_duration_sec))
            kept += 1
            stats['kept'] += 1
            yield sample
        elif active and ((max_duration_sec is not None and dur > max_duration_sec) or
                         (min_duration_sec is not None and dur < min_duration_sec)):
            dropped += 1
            dropped_sec += dur
            stats['dropped'] += 1
            stats['dropped_sec'] += dur
        else:
            kept += 1
            kept_sec += dur
            stats['kept'] += 1
            stats['kept_sec'] += dur
            yield sample
        if log_every and seen % log_every == 0:
            _line('after {} rows'.format(seen), kept, dropped, kept_sec, dropped_sec)
    _line('epoch end', kept, dropped, kept_sec, dropped_sec)


def budget_check(data, context_limit=CONTEXT_LIMIT, margin=SAFETY_MARGIN, mode='train'):
    """Validate every sample against PLAN §11.2. Raises instead of dropping.

    Also removes `audio_data` (not needed by the LLM) to keep the worker RAM small.
    """
    for sample in data:
        assert 'text_token' in sample and 'speech_token' in sample, sample.get('utt')
        n_text = len(sample['text_token'])
        n_instr = len(sample.get('instruct_token', []))
        n_speech = len(sample['speech_token'])
        if n_text == 0 or n_speech == 0:
            raise ContextBudgetError('empty text/speech tokens for utt {} (n_text={}, n_speech={}); '
                                     'prepare_arms must have excluded it explicitly'.format(sample.get('utt'), n_text, n_speech))
        b = budget_row(n_text, n_speech, n_instr, context_limit, margin)
        if not b['fits']:
            raise ContextBudgetError('context budget violated by utt {}: seq_max {} > budget {} (text {}, speech {}, instr {})'
                                     .format(sample.get('utt'), b['seq_max'], b['budget'], n_text, n_speech, n_instr))
        sample['n_speech'] = n_speech
        sample['seq_len'] = b['seq_max']
        sample.pop('audio_data', None)
        yield sample


def sort_by_tokens(data, sort_size=500, mode='train'):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['n_speech'])
            for x in buf:
                yield x
            buf = []
    buf.sort(key=lambda x: x['n_speech'])
    for x in buf:
        yield x


def token_batch(data, max_speech_tokens_in_batch=22500, max_padded_positions=28672, mode='train'):
    """Token-based dynamic batch (PLAN §11.3).

    Both caps are always supplied by configs/train/cv3_{long,short}_sft.yaml; the defaults here only
    mirror the values frozen by the Lead decision of 2026-08-28 (long cap 22500 / short cap 20700 --
    the short cap was corrected from 17000 on 2026-08-29 after the measured long-arm mean of 20060,
    reports/training_budget_comparison.md 4.1 -- max_padded_positions 28672 for both arms) so that a
    direct call cannot silently use the old 32768.

    A batch is closed BEFORE adding a sample if either
      sum(target speech tokens) + sample.n_speech > max_speech_tokens_in_batch, or
      max(seq_len) * (len(batch) + 1) > max_padded_positions  (memory guard: padded LM positions).
    A single sample always forms a valid batch even if its own seq_len exceeds max_padded_positions
    (the guard is only checked when the buffer is non-empty): the cap limits padding waste ACROSS
    samples and must never drop or truncate one (PLAN §4.2/§11.2 - budget_check already raised for
    anything above context_limit - margin = 31744).
    """
    buf, tokens, longest = [], 0, 0
    for sample in data:
        n, L = sample['n_speech'], sample['seq_len']
        new_longest = max(longest, L)
        if buf and (tokens + n > max_speech_tokens_in_batch or new_longest * (len(buf) + 1) > max_padded_positions):
            yield buf
            buf, tokens, longest = [sample], n, L
        else:
            buf.append(sample)
            tokens += n
            longest = new_longest
    if buf:
        yield buf


def padding_llm(data, mode='train', gan=False, dpo=False):
    """Collate for CosyVoice3LM.forward: text_token(+len), speech_token(+len), instruct_token(+len), utts, text.

    Extra keys (ignored by the model, read by the executor for logging): n_speech_tokens (sum of target
    tokens in the batch), n_text_tokens, padded_positions (= longest seq_len x batch size), batch_size.
    """
    assert gan is False and dpo is False, 'processor_ext.padding_llm is for LLM SFT only'
    for samples in data:
        assert isinstance(samples, list)
        order = torch.argsort(torch.tensor([x['n_speech'] for x in samples], dtype=torch.int32), descending=True)
        samples = [samples[i] for i in order]
        batch = {'utts': [x['utt'] for x in samples], 'text': [x['text'] for x in samples]}
        text_token = [torch.tensor(x['text_token'], dtype=torch.int64) for x in samples]
        batch['text_token_len'] = torch.tensor([t.size(0) for t in text_token], dtype=torch.int32)
        batch['text_token'] = pad_sequence(text_token, batch_first=True, padding_value=0)
        speech_token = [torch.as_tensor(x['speech_token'], dtype=torch.int64) for x in samples]
        batch['speech_token_len'] = torch.tensor([t.size(0) for t in speech_token], dtype=torch.int32)
        batch['speech_token'] = pad_sequence(speech_token, batch_first=True, padding_value=0)
        if all('instruct_token' in x for x in samples):
            instruct_token = [torch.tensor(x['instruct_token'], dtype=torch.int64) for x in samples]
            batch['instruct_token_len'] = torch.tensor([t.size(0) for t in instruct_token], dtype=torch.int32)
            batch['instruct_token'] = pad_sequence(instruct_token, batch_first=True, padding_value=0)
        else:
            raise ContextBudgetError('instruct_token missing for {}'.format(batch['utts']))
        batch['n_speech_tokens'] = int(batch['speech_token_len'].sum())
        batch['n_text_tokens'] = int(batch['text_token_len'].sum())
        batch['padded_positions'] = int(max(x['seq_len'] for x in samples) * len(samples))
        batch['batch_size'] = len(samples)
        yield batch


def stats_logger(data, log_every=0, mode='train'):
    """Optional: log batch composition for the first batches (debug)."""
    n = 0
    for batch in data:
        n += 1
        if log_every and n <= log_every:
            logging.info('batch %d: size %d speech_tokens %d text_tokens %d padded_positions %d', n, batch['batch_size'],
                         batch['n_speech_tokens'], batch['n_text_tokens'], batch['padded_positions'])
        yield batch

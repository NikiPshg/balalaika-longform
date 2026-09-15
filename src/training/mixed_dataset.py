"""Step-level source mixing for the LLM-only SFT (E6 Mixed-SFT recipe, A6).

Pre-registered by the Lead in reports/decisions.md, 2026-08-30 ("ПРЕРЕГИСТРАЦИЯ E6 «Mixed-SFT recipe»"):
every optimizer step consumes ONE batch drawn entirely from ONE source (S = ext_short, L = long); the
sources follow a cyclic pattern (E6-S = `S`, E6-M31 = `S,S,S,L`, E6-M11 = `S,L`, E6-M13 = `S,L,L,L`); each
source keeps its OWN token cap (L 22500 / S 20700 -- the frozen E3/E2 caps, so a mixed step costs the same
target tokens as an E3 or an E2 step); when a source is exhausted it restarts with epoch+1 (a new shard
shuffle); the run is bounded by --max_steps, not by epochs.

What this module provides
  parse_pattern     'S,S,S,L' -> ('S', 'S', 'S', 'L'); raises on an empty pattern
  with_token_cap    copy of the yaml data_pipeline whose token_batch stage gets ANOTHER
                    max_speech_tokens_in_batch (the rest of the stages are the same objects)
  MixedDataset      IterableDataset over {name: per-source IterableDataset}; cycles the pattern, tags every
                    batch with `source` / `source_epoch`, restarts an exhausted source with epoch+1
  build_mixed_dataset / init_mixed_dataset_and_dataloader
                    the config-driven constructors used by src/training/cosyvoice_train/train.py when
                    --mix_pattern is given (and ONLY then: the default code path is untouched)

Per-source pipelines. Each source is the stock cosyvoice.dataset.dataset.Dataset(...) over its own
data.list with the SAME data_pipeline as configs/train/cv3_long_sft.yaml (parquet_opener -> tokenize ->
budget_check -> shuffle -> sort_by_tokens -> token_batch -> padding_llm) except that token_batch gets the
source's cap from the yaml block `mix_sources: {L: {max_speech_tokens_in_batch: 22500}, S: {...: 20700}}` and,
for S, its own sort buffer `sort_size: 3000` (Lead amendment 2, 2026-08-30: with the yaml 500 a ~110-clip
ext_short batch straddles a sort chunk every 4-5 batches and closes at 4-9k tokens; S mean 17.4k -> 20.0k).
The VRAM guard max_padded_positions stays the yaml 28672 for both sources and the cv pass (a per-source
override is supported but unused: amendment 1, S 34000, was measured insufficient and superseded).
The dev (cv) pass is NOT mixed: train.py builds it exactly as before from --cv_data and the unmodified
data_pipeline (long cap), so the dev loss stays comparable with E2/E3/E4.

Workers and the global batch sequence (num_workers = N, PLAN §0.1 uses N = 2). The DataLoader pickles the
dataset into every worker; each worker therefore runs its OWN copy of the cyclic pattern and its OWN
per-source pipelines, and the stock DistributedSampler gives worker w the shards `w::N` of every source
(after the epoch-seeded shard shuffle). The DataLoader hands batches back in strict round-robin worker
order (w0, w1, w0, w1, ...), so the GLOBAL sequence seen by the optimizer is the pattern with every element
repeated N times: for `S,S,S,L` and N = 2 it is S,S,S,S,S,S,L,L,S,S,S,S,S,S,L,L,... The source proportions
are exact over every len(pattern) x N steps and the per-source counts of any prefix differ from the ideal
pattern ratio by less than N (i.e. "+-1 per worker"); the ORDER is the pattern only approximately. This
is documented in the E6 pre-registration and re-measured in the smoke (reports/decisions.md).

Restart on exhaustion. A source's iterator ends when the worker has consumed all of its shards for the
current epoch. The dataset then calls `set_epoch(epoch + 1)` on THAT source (a new shard shuffle; the
in-pipeline `shuffle` buffer is re-randomised anyway), re-iterates it and continues the pattern without a
gap. Every restart is logged with the worker id, the number of batches that source yielded in the finished
epoch and the total number of batches this worker has yielded so far (the optimizer step is not known
inside a worker; with N workers it is ~N x that count). Each worker restarts its sources independently, so
the two workers may be in different epochs of the same source; nothing is skipped -- every shard of every
epoch a worker starts is consumed unless --max_steps stops the run first.

Nothing is dropped silently: a source whose data.list is empty, or whose first pass yields no batch at
all (e.g. every shard failed to open), raises MixSourceError instead of spinning; a pattern that names an
unknown source raises at construction.
"""
import logging
import os
from functools import partial

import torch
from torch.utils.data import DataLoader, IterableDataset

# The stage of the yaml data_pipeline whose keyword carries the per-source cap (src/training/processor_ext.py).
TOKEN_BATCH_STAGE = 'token_batch'
CAP_KEY = 'max_speech_tokens_in_batch'
POS_KEY = 'max_padded_positions'      # OPTIONAL per-source VRAM guard; no source sets it after Lead amendment 2
                                      # (2026-08-30): both inherit the yaml 28672 (amendment 1, S 34000, superseded)
SORT_STAGE = 'sort_by_tokens'
SORT_KEY = 'sort_size'                # OPTIONAL per-source sort buffer; Lead amendment 2 (2026-08-30): S 3000.
                                      # A 500-row sort chunk is only ~4-5 ext_short batches wide, so a batch straddling
                                      # a chunk boundary inherits a long `longest` and closes at 4-9k tokens (S mean
                                      # 17.4k even at guard 34000); with 3000 the 20700 token cap binds (S mean 20.0k)
CONFIG_KEY = 'mix_sources'


class MixSourceError(RuntimeError):
    pass


def parse_pattern(pattern):
    """'S,S,S,L' (or an iterable of names) -> tuple of non-empty source names, in order."""
    if isinstance(pattern, str):
        names = [p.strip() for p in pattern.split(',')]
    else:
        names = [str(p).strip() for p in pattern]
    names = tuple(n for n in names if n)
    if not names:
        raise MixSourceError('empty mix pattern {!r}'.format(pattern))
    return names


def pattern_counts(pattern):
    """{name: occurrences in one cycle of the pattern}."""
    out = {}
    for n in parse_pattern(pattern):
        out[n] = out.get(n, 0) + 1
    return out


def _stage_name(stage):
    f = stage.func if isinstance(stage, partial) else stage
    return getattr(f, '__name__', None)


def _token_batch_index(data_pipeline):
    hits = [i for i, st in enumerate(data_pipeline) if _stage_name(st) == TOKEN_BATCH_STAGE]
    if len(hits) != 1:
        raise MixSourceError('data_pipeline must contain exactly one {} stage, found {}'.format(TOKEN_BATCH_STAGE, len(hits)))
    return hits[0]


def pipeline_token_batch_keywords(data_pipeline):
    """The keywords the yaml gave the token_batch stage (cap and guard used by the cv pass / non-mixed runs)."""
    st = data_pipeline[_token_batch_index(data_pipeline)]
    return dict(st.keywords or {}) if isinstance(st, partial) else {}


def _with_keywords(stage, **kw):
    if isinstance(stage, partial):
        merged = dict(stage.keywords or {})
        merged.update(kw)
        return partial(stage.func, *stage.args, **merged)
    return partial(stage, **kw)


def with_token_cap(data_pipeline, cap, max_padded_positions=None, sort_size=None):
    """Copy of `data_pipeline` where the single token_batch stage gets max_speech_tokens_in_batch=cap and,
    when `max_padded_positions` is given, that VRAM guard too (Lead correction 2026-08-30: S 34000); when
    `sort_size` is given, the single sort_by_tokens stage gets that buffer size (optional, see SORT_KEY).

    Every other stage is the SAME object (the partials the yaml built). The original list is not modified,
    so the cv pass, which is built from configs['data_pipeline'], keeps the yaml cap, guard and sort buffer.
    """
    i = _token_batch_index(data_pipeline)
    out = list(data_pipeline)
    kw = {CAP_KEY: int(cap)}
    if max_padded_positions is not None:
        kw[POS_KEY] = int(max_padded_positions)
    out[i] = _with_keywords(out[i], **kw)
    if sort_size is not None:
        hits = [j for j, st in enumerate(out) if _stage_name(st) == SORT_STAGE]
        if len(hits) != 1:
            raise MixSourceError('data_pipeline must contain exactly one {} stage to override {}, found {}'.format(SORT_STAGE, SORT_KEY, len(hits)))
        out[hits[0]] = _with_keywords(out[hits[0]], **{SORT_KEY: int(sort_size)})
    return out


def _worker_tag():
    info = torch.utils.data.get_worker_info()
    if info is None:
        return 'pid {} worker 0/1'.format(os.getpid())
    return 'pid {} worker {}/{}'.format(os.getpid(), info.id, info.num_workers)


class MixedDataset(IterableDataset):
    """Cycle `pattern` over `sources` ({name: IterableDataset with set_epoch}); one batch per element.

    Every yielded batch is the source's own batch dict (as produced by its pipeline, e.g. the
    processor_ext.padding_llm collate) plus two keys: `source` (the name) and `source_epoch` (the
    epoch of that source the batch came from, starting at the epoch given to set_epoch()).
    The iterator is INFINITE by design (the trainer stops at --max_steps); a source that yields nothing in
    a whole pass raises MixSourceError.
    """

    def __init__(self, sources, pattern):
        if not sources:
            raise MixSourceError('no mix sources given')
        self.sources = dict(sources)
        self.pattern = parse_pattern(pattern)
        unknown = sorted(set(self.pattern) - set(self.sources))
        if unknown:
            raise MixSourceError('pattern {} names unknown source(s) {}; known: {}'.format(
                ','.join(self.pattern), unknown, sorted(self.sources)))
        for name, src in self.sources.items():
            if not hasattr(src, 'set_epoch'):
                raise MixSourceError('source {!r} has no set_epoch(); a stock cosyvoice Dataset is expected'.format(name))
        self.epoch = -1
        self.counts = pattern_counts(self.pattern)

    def set_epoch(self, epoch):
        """Base epoch for every source (the trainer's outer epoch; the per-source counters start here)."""
        self.epoch = int(epoch)
        for src in self.sources.values():
            src.set_epoch(self.epoch)

    def __iter__(self):
        tag = _worker_tag()
        used = [n for n in self.sources if n in self.counts]
        epoch = {n: self.epoch for n in used}
        iters = {}
        yielded_epoch = {n: 0 for n in used}     # batches of the current epoch of that source
        yielded_total = {n: 0 for n in used}
        total = 0
        for n in used:
            self.sources[n].set_epoch(epoch[n])
            iters[n] = iter(self.sources[n])
        logging.info('mixed_dataset[%s] pattern %s, sources %s, start epoch %d', tag, ','.join(self.pattern),
                     {n: self.counts[n] for n in used}, self.epoch)
        pos = 0
        while True:
            name = self.pattern[pos % len(self.pattern)]
            pos += 1
            try:
                batch = next(iters[name])
            except StopIteration:
                if yielded_epoch[name] == 0:
                    raise MixSourceError('mix source {!r} yielded no batch in epoch {} ({}): empty data.list, '
                                         'unreadable shards or a filter that dropped everything'.format(name, epoch[name], tag))
                logging.info('mixed_dataset[%s] source %s exhausted: %d batches in epoch %d (%d total from this source, '
                             '%d batches yielded by this worker so far); restarting with epoch %d',
                             tag, name, yielded_epoch[name], epoch[name], yielded_total[name], total, epoch[name] + 1)
                epoch[name] += 1
                yielded_epoch[name] = 0
                self.sources[name].set_epoch(epoch[name])
                iters[name] = iter(self.sources[name])
                try:
                    batch = next(iters[name])
                except StopIteration:
                    raise MixSourceError('mix source {!r} yielded no batch right after its restart (epoch {}, {})'
                                         .format(name, epoch[name], tag))
            if not isinstance(batch, dict):
                raise MixSourceError('mix source {!r} must yield collated batch dicts, got {}'.format(name, type(batch).__name__))
            batch['source'] = name
            batch['source_epoch'] = epoch[name]
            yielded_epoch[name] += 1
            yielded_total[name] += 1
            total += 1
            yield batch


def resolve_mix_sources(configs, pattern, cli_sources=None):
    """{name: {'data_list': path, 'cap': int}} for every source the pattern uses.

    Caps come from configs['mix_sources'][name]['max_speech_tokens_in_batch'] (the yaml is the single place
    where the frozen per-source caps are written; tests/test_training_configs.py asserts them). The optional
    per-source `max_padded_positions` overrides the VRAM guard of that source's token_batch (unused since Lead
    amendment 2; when absent the source keeps the yaml value 28672, as the cv pass does); the optional per-source
    `sort_size` overrides that source's sort_by_tokens buffer (amendment 2: S 3000). The data.list comes from
    the CLI (`--mix_source NAME=path`, wins) or from the yaml block's `data_list`.
    """
    names = parse_pattern(pattern)
    default_pos = pipeline_token_batch_keywords(configs['data_pipeline']).get(POS_KEY) if configs.get('data_pipeline') else None
    block = configs.get(CONFIG_KEY)
    if not isinstance(block, dict) or not block:
        raise MixSourceError('the config has no `{}` block; --mix_pattern needs it (configs/train/cv3_mix_sft.yaml)'.format(CONFIG_KEY))
    cli = {}
    for spec in cli_sources or []:
        if '=' not in spec:
            raise MixSourceError('--mix_source expects NAME=path, got {!r}'.format(spec))
        n, p = spec.split('=', 1)
        n, p = n.strip(), p.strip()
        if not n or not p:
            raise MixSourceError('--mix_source expects NAME=path, got {!r}'.format(spec))
        if n in cli:
            raise MixSourceError('--mix_source {} given twice'.format(n))
        cli[n] = p
    unknown = sorted(set(cli) - set(block))
    if unknown:
        raise MixSourceError('--mix_source names {} that are not in the config `{}` block {}'.format(unknown, CONFIG_KEY, sorted(block)))
    out = {}
    for n in sorted(set(names)):
        if n not in block:
            raise MixSourceError('pattern {} uses source {!r} which the config `{}` block does not define ({})'.format(
                ','.join(names), n, CONFIG_KEY, sorted(block)))
        entry = block[n] or {}
        if CAP_KEY not in entry:
            raise MixSourceError('config {}.{} has no {}'.format(CONFIG_KEY, n, CAP_KEY))
        data_list = cli.get(n) or entry.get('data_list')
        if not data_list:
            raise MixSourceError('no data.list for mix source {!r}: pass --mix_source {}=<path> or set {}.{}.data_list'.format(n, n, CONFIG_KEY, n))
        if not os.path.exists(data_list):
            raise MixSourceError('mix source {!r}: data.list not found: {}'.format(n, data_list))
        with open(data_list, encoding='utf-8') as f:
            n_lines = sum(1 for line in f if line.strip())
        if n_lines == 0:
            raise MixSourceError('mix source {!r}: data.list is empty: {}'.format(n, data_list))
        pos = entry.get(POS_KEY, default_pos)
        sort_size = entry.get(SORT_KEY)
        out[n] = {'data_list': data_list, 'cap': int(entry[CAP_KEY]), 'n_shards': n_lines,
                  POS_KEY: int(pos) if pos is not None else None,
                  SORT_KEY: int(sort_size) if sort_size is not None else None}
    return out


def build_mixed_dataset(configs, pattern, cli_sources=None, mode='train', gan=False, dpo=False):
    """MixedDataset over per-source stock Datasets built from the yaml data_pipeline with per-source caps.

    Returns (dataset, info) where info describes the resolved sources (for run_info.json).
    """
    from cosyvoice.dataset.dataset import Dataset  # read-only upstream, imported lazily (torch/torchaudio)
    resolved = resolve_mix_sources(configs, pattern, cli_sources)
    sources = {}
    for name, r in resolved.items():
        pipeline = with_token_cap(configs['data_pipeline'], r['cap'], r[POS_KEY], r[SORT_KEY])
        sources[name] = Dataset(r['data_list'], data_pipeline=pipeline, mode=mode, gan=gan, dpo=dpo, shuffle=True, partition=True)
        logging.info('mix source %s: data_list %s (%d shards), %s %d, %s %s, %s %s', name, r['data_list'], r['n_shards'],
                     CAP_KEY, r['cap'], POS_KEY, r[POS_KEY], SORT_KEY, r[SORT_KEY] if r[SORT_KEY] is not None else 'yaml')
    ds = MixedDataset(sources, pattern)
    info = {'pattern': ','.join(ds.pattern), 'pattern_counts': ds.counts, 'sources': resolved}
    return ds, info


def init_mixed_dataset_and_dataloader(args, configs, gan, dpo):
    """Mixed-mode replacement for cosyvoice.utils.train_utils.init_dataset_and_dataloader.

    Train = MixedDataset (pattern args.mix_pattern, lists args.mix_source / yaml); cv = the stock
    Dataset(args.cv_data, configs['data_pipeline'], mode='dev', shuffle=False, partition=False), i.e. byte-
    for-byte what the default path builds. The DataLoader arguments are the stock ones.
    """
    from cosyvoice.dataset.dataset import Dataset
    assert gan is False and dpo is False, 'mixing is implemented for the LLM SFT only'
    data_pipeline = configs['data_pipeline']
    train_dataset, info = build_mixed_dataset(configs, args.mix_pattern, args.mix_source, mode='train', gan=gan, dpo=dpo)
    cv_dataset = Dataset(args.cv_data, data_pipeline=data_pipeline, mode='dev', gan=gan, dpo=dpo, shuffle=False, partition=False)
    train_data_loader = DataLoader(train_dataset, batch_size=None, pin_memory=args.pin_memory,
                                   num_workers=args.num_workers, prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset, batch_size=None, pin_memory=args.pin_memory,
                                num_workers=args.num_workers, prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader, info

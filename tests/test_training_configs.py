"""A6: the two arm configs are identical below their one-line header EXCEPT exactly one line.

PLAN §6.4 fixes one init checkpoint, one set of trainable modules, one optimizer/schedule and one
number of optimizer steps for E2 (short) and E3 (long); "any mismatch is documented in advance".
The Lead decision of 2026-08-28 (reports/decisions.md, "ПРИНЯТО PROPOSED A6") introduced the single
intended mismatch: budget-matching variant (b), token cap 22500 for the long arm and 20700 (corrected 2026-08-29 from 17000 after the measured long mean 20060) for the
short arm, so that an equal number of optimizer steps gives equal MEAN target tokens/step within +-3 %
(verified on the first 200 pilot steps with scripts/check_budget_match.py).

This test therefore asserts
  * exactly ONE differing line below the header, and that line is token_batch.max_speech_tokens_in_batch;
  * the frozen values of that decision: 22500 / 20700, max_padded_positions 28672 in both arms,
    llm.mix_ratio [5, 1000000] in both arms;
  * the other frozen recipe values (context limit 32768, margin 1024, accum_grad 1, the A6 processors),
so that a silent edit of any of them fails the suite instead of quietly changing the comparison.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LONG_CFG = 'cv3_long_sft.yaml'
SHORT_CFG = 'cv3_short_sft.yaml'

# Frozen by the Lead decision of 2026-08-28 (reports/decisions.md). Changing any of these needs a new
# decision line there AND an update of this test - that is the point of the test.
DECISION_DATE = '2026-08-28'
CAP_KEY = 'max_speech_tokens_in_batch'
FROZEN_CAP = {LONG_CFG: '22500', SHORT_CFG: '20700'}
FROZEN_BOTH = {
    'max_padded_positions': '28672',   # was 32768; a 15-min unit needs 28087 padded positions
    'mix_ratio': '[5, 1000000]',       # unistream-only for both arms
    'context_limit': '32768',          # CosyVoice-BlankEN max_position_embeddings
    'margin': '1024',                  # frozen context-budget safety margin
}


def _read(name):
    with open(os.path.join(ROOT, 'configs', 'train', name), encoding='utf-8') as f:
        return f.read().splitlines()


def _body(name):
    """Everything below the one-line arm header (line 0 is the only line allowed to name the arm)."""
    lines = _read(name)
    assert lines[0].startswith('# A6 SFT recipe'), lines[0]
    return lines[1:]


def _section(name, section):
    """Lines of one top-level yaml block (e.g. `train_conf:`), so that the unused gan blocks cannot match."""
    lines, out, inside = _body(name), [], False
    for line in lines:
        if line.startswith(section + ':'):
            inside = True
            continue
        if inside:
            if line and not line[0].isspace() and not line.startswith('#'):
                break
            out.append(line)
    assert out, '{}: no `{}:` block'.format(name, section)
    return out


def _scalar(name, key, section=None, indent=None):
    """Value of a yaml scalar key, ignoring comment lines and trailing comments. Must occur exactly once.

    `indent` pins the nesting level (train_conf has both `warmup_steps: 50` at depth 1 and the stock
    `scheduler_conf.warmup_steps: 2500` at depth 2).
    """
    lines = _section(name, section) if section else _body(name)
    depth = r'\s*' if indent is None else ' ' * indent
    pat = re.compile(r'^' + depth + re.escape(key) + r':\s*(.*?)\s*(?:#.*)?$')
    hits = [m.group(1) for m in (pat.match(l) for l in lines) if m]
    assert len(hits) == 1, '{}: expected exactly one `{}:` line{}, found {}'.format(
        name, key, ' in ' + section if section else '', len(hits))
    return hits[0]


def test_arm_configs_differ_only_in_the_token_cap():
    long_body, short_body = _body(LONG_CFG), _body(SHORT_CFG)
    assert len(long_body) == len(short_body), 'configs have different line counts'
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(long_body, short_body)) if a != b]
    assert len(diff) == 1, 'expected exactly one differing line below the header, got {}'.format(
        [(i + 2, a, b) for i, a, b in diff])
    _, a, b = diff[0]
    assert a.strip().startswith(CAP_KEY + ':'), 'the only allowed difference is {}, got: {!r}'.format(CAP_KEY, a)
    assert b.strip().startswith(CAP_KEY + ':'), 'the only allowed difference is {}, got: {!r}'.format(CAP_KEY, b)


@pytest.mark.parametrize('cfg', [LONG_CFG, SHORT_CFG])
def test_token_cap_frozen(cfg):
    assert _scalar(cfg, CAP_KEY) == FROZEN_CAP[cfg]


@pytest.mark.parametrize('cfg', [LONG_CFG, SHORT_CFG])
@pytest.mark.parametrize('key,value', sorted(FROZEN_BOTH.items()))
def test_values_frozen_in_both_arms(cfg, key, value):
    assert _scalar(cfg, key) == value


@pytest.mark.parametrize('cfg', [LONG_CFG, SHORT_CFG])
def test_recipe_invariants(cfg):
    body = '\n'.join(_body(cfg))
    assert 'src.training.processor_ext.budget_check' in body      # raises, never drops (PLAN §4.2/§11.2)
    assert 'src.training.processor_ext.token_batch' in body       # token-based batching (PLAN §11.3)
    assert 'src.training.processor_ext.padding_llm' in body
    # train_conf only: the *_gan blocks below it are never used with --model llm and carry the same keys
    assert _scalar(cfg, 'accum_grad', 'train_conf') == '1'        # micro-batch == effective batch
    assert _scalar(cfg, 'lr', 'train_conf', indent=8) == '1e-5'
    assert _scalar(cfg, 'warmup_steps', 'train_conf', indent=4) == '50'   # not scheduler_conf.warmup_steps
    assert _scalar(cfg, 'save_per_step', 'train_conf') == '200'
    assert _scalar(cfg, 'grad_clip', 'train_conf') == '5'


@pytest.mark.parametrize('cfg', [LONG_CFG, SHORT_CFG])
def test_header_documents_the_decision(cfg):
    """The header must carry the decisions.md date, otherwise the asymmetry is undocumented."""
    header = '\n'.join(l for l in _read(cfg) if l.startswith('#'))
    assert DECISION_DATE in header, '{}: header does not cite the decisions.md date {}'.format(cfg, DECISION_DATE)
    for value in (FROZEN_CAP[LONG_CFG], FROZEN_CAP[SHORT_CFG], FROZEN_BOTH['max_padded_positions']):
        assert value in header, '{}: header does not document the frozen value {}'.format(cfg, value)


def test_headers_name_the_two_arms():
    assert 'E3 Long-SFT' in _read(LONG_CFG)[0]
    assert 'E2 Short-SFT' in _read(SHORT_CFG)[0]


# ---------------------------------------------------------------------------------------------------
# E4 Curriculum-SFT (A6, 2026-08-29). The Lead pre-registered in reports/decisions.md three stages on the
# LONG arm with a growing ceiling on the duration of one training unit: S1 <= 90 s (500 steps),
# S2 <= 180 s (500 steps), S3 <= 900 s (2000 steps, the whole arm -- a no-op kept explicit). The stage
# configs must therefore be the long config and NOTHING else, plus the duration_filter that implements the
# ceiling: any other drift between a stage and E3 would make the E4 row incomparable with E3/E2.
#
# The property is checked structurally: each stage config is an E4 header block, terminated by
# E4_HEADER_END, followed by cv3_long_sft.yaml verbatim with a few inserted lines, and EVERY inserted line
# carries the marker E4_TAG. Strip the header, strip the marked lines, and what remains must equal
# cv3_long_sft.yaml line for line.
# ---------------------------------------------------------------------------------------------------
E4_TAG = 'E4-CURRICULUM'
E4_HEADER_END = '# ==== END E4 CURRICULUM HEADER'
E4_DECISION_DATE = '2026-08-29'
# Pre-registered ceilings, seconds. Changing one needs a new line in reports/decisions.md.
CURRICULUM = {'cv3_curriculum_s1.yaml': '90', 'cv3_curriculum_s2.yaml': '180', 'cv3_curriculum_s3.yaml': '900'}


def _split_curriculum(name):
    """(header lines incl. the end marker, unmarked body lines, marked lines) of a stage config."""
    lines = _read(name)
    idx = [i for i, l in enumerate(lines) if l.startswith(E4_HEADER_END)]
    assert len(idx) == 1, '{}: expected exactly one {!r} line, found {}'.format(name, E4_HEADER_END, len(idx))
    header, rest = lines[:idx[0] + 1], lines[idx[0] + 1:]
    return header, [l for l in rest if E4_TAG not in l], [l for l in rest if E4_TAG in l]


def test_long_config_carries_no_curriculum_marker():
    """The marker-stripping rule below is only sound if the long config never contains the marker."""
    assert E4_TAG not in '\n'.join(_read(LONG_CFG))
    assert E4_HEADER_END not in '\n'.join(_read(LONG_CFG))


@pytest.mark.parametrize('cfg', sorted(CURRICULUM))
def test_curriculum_config_is_the_long_config_plus_only_the_duration_filter(cfg):
    header, body, marked = _split_curriculum(cfg)
    assert marked, '{}: no {} lines, so the stage has no ceiling'.format(cfg, E4_TAG)
    long_lines = _read(LONG_CFG)
    assert len(body) == len(long_lines), (
        '{}: below the header and outside the {} lines it must be cv3_long_sft.yaml verbatim, but it has '
        '{} lines against {}'.format(cfg, E4_TAG, len(body), len(long_lines)))
    diff = [(i + 1, a, b) for i, (a, b) in enumerate(zip(body, long_lines)) if a != b]
    assert not diff, '{}: differs from cv3_long_sft.yaml outside the header and the {} lines: {}'.format(
        cfg, E4_TAG, diff[:5])


@pytest.mark.parametrize('cfg,seconds', sorted(CURRICULUM.items()))
def test_curriculum_ceiling_is_the_preregistered_one(cfg, seconds):
    _, _, marked = _split_curriculum(cfg)
    hits = [m.group(1) for m in (re.match(r'^\s*max_duration_sec:\s*(\S+?)\s*(?:#.*)?$', l) for l in marked) if m]
    assert hits == [seconds], '{}: expected exactly one `max_duration_sec: {}`, found {}'.format(cfg, seconds, hits)


@pytest.mark.parametrize('cfg', sorted(CURRICULUM))
def test_curriculum_marked_lines_are_only_the_duration_filter(cfg):
    """Nothing but the filter may hide behind the marker, otherwise the verbatim check could be defeated."""
    _, _, marked = _split_curriculum(cfg)
    for line in marked:
        assert any(k in line for k in ('duration_filter', 'max_duration_sec', 'log_every')), \
            '{}: marked line is not part of the duration_filter block: {!r}'.format(cfg, line)
    assert sum(1 for l in marked if l.startswith('duration_filter: !name:src.training.processor_ext.duration_filter')) == 1
    assert sum(1 for l in marked if l.split('#')[0].strip().rstrip(',') == '!ref <duration_filter>') == 1


@pytest.mark.parametrize('cfg', sorted(CURRICULUM))
def test_curriculum_filter_is_the_first_pipeline_stage_after_parquet_opener(cfg):
    """Before tokenize: a row the ceiling rejects must not be tokenized, and it must never reach
    budget_check/token_batch, whose counts feed the budget bookkeeping."""
    lines = _read(cfg)
    i = lines.index('data_pipeline: [')
    seq = [l.split('#')[0].strip().rstrip(',') for l in lines[i + 1:i + 4]]
    assert seq == ['!ref <parquet_opener>', '!ref <duration_filter>', '!ref <tokenize>'], seq


@pytest.mark.parametrize('cfg', sorted(CURRICULUM))
def test_curriculum_header_cites_the_preregistration(cfg):
    header, _, _ = _split_curriculum(cfg)
    for line in header:
        assert not line.strip() or line.startswith('#'), '{}: non-comment line in the header: {!r}'.format(cfg, line)
    text = '\n'.join(header)
    assert E4_DECISION_DATE in text, '{}: header does not cite the decisions.md date {}'.format(cfg, E4_DECISION_DATE)
    assert 'decisions.md' in text and 'E4' in text and 'Curriculum' in text
    # the whole pre-registered schedule is documented in every stage, not just the stage's own ceiling
    for value in ('90 s', '180 s', '900 s', '500 steps', '2000 steps'):
        assert value in text, '{}: header does not document {!r}'.format(cfg, value)
    # and the reason the dev pass is NOT filtered (comparable dev loss across stages, PLAN §11.5)
    assert "mode='dev'" in text and 'dev loss' in text


@pytest.mark.parametrize('cfg', sorted(CURRICULUM))
def test_curriculum_keeps_the_frozen_long_arm_values(cfg):
    """Redundant with the verbatim check, but names the values so a failure says which one moved."""
    body = _split_curriculum(cfg)[1]
    text = '\n'.join(body)
    assert 'max_speech_tokens_in_batch: {}'.format(FROZEN_CAP[LONG_CFG]) in text
    assert 'max_padded_positions: {}'.format(FROZEN_BOTH['max_padded_positions']) in text
    assert 'mix_ratio: {}'.format(FROZEN_BOTH['mix_ratio']) in text
    assert 'warmup_steps: 50' in text and 'save_per_step: 200' in text and 'lr: 1e-5' in text


# ---------------------------------------------------------------------------------------------------
# E4 wiring: the runner and the trainer must implement the pre-registered schedule, not just the ceilings.
# A CosyVoice checkpoint carries `step`/`epoch` scalars inside the .pt (train_utils.save_model), and
# train.py resumes its counters from them. That is right for a continued run and fatal for a curriculum
# stage: S1 initialised from E2's checkpoint at step ~2600 with --max_steps 500 would stop before taking a
# single step, and S2/S3 would inherit a warmup that is already over. Hence `-- --reset_steps` per stage.
# ---------------------------------------------------------------------------------------------------
RUNNER = os.path.join(ROOT, 'scripts', 'run_curriculum.sh')
TRAIN_PY = os.path.join(ROOT, 'src', 'training', 'cosyvoice_train', 'train.py')


def _text(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def test_trainer_can_restart_the_step_counter():
    src = _text(TRAIN_PY)
    assert "'--reset_steps'" in src, 'train.py lost the --reset_steps option'
    assert 'action=\'store_true\', default=False' in src.split("'--reset_steps'")[1][:120], \
        '--reset_steps must default to off so E2/E3 behaviour is unchanged'
    tail = src.split('model.load_state_dict(state_dict, strict=False)')[1][:1200]
    assert 'if args.reset_steps:' in tail, 'the step/epoch inheritance is not guarded by --reset_steps'
    assert tail.index('if args.reset_steps:') < tail.index("if 'step' in state_dict:"), \
        'the guard must come before the step/epoch inheritance'


def test_runner_launches_the_preregistered_schedule():
    src = _text(RUNNER)
    assert 'STAGES=(s1 s2 s3)' in src
    assert 'STEPS=(500 500 2000)' in src, 'the pre-registered step budget is 500 / 500 / 2000'
    assert 'CEIL=(90 180 900)' in src
    assert '--exp_dir exp/curriculum' in src, 'stages must not land in exp/long'
    # match the executed line, not the header comment that also mentions the flag
    assert '-- --reset_steps > "$LOG/train_$st.log"' in src, \
        'the train_arm.sh invocation must forward --reset_steps so each stage restarts its counter/warmup'
    assert 'configs/train/cv3_curriculum_${st}.yaml' in src
    assert '--keep_best 1' in src, 'keep best + last per stage'
    assert '--experiment E4' in src and "--voice-filter 'ref_(female|male)_01$'" in src
    assert '--floor-cache results/$BASE/asr_floor.jsonl' in src
    for label in ('E1_native=results/$BASE/E1', 'E0_official=results/$BASE/E0'):
        assert label in src
    assert 'results/$SFT/E2_* results/$SFT/E3_*' in src, 'the pilot E2/E3 rows must join the main table'


# ---------------------------------------------------------------------------------------------------
# E6 Mixed-SFT (A6, 2026-08-30). The Lead pre-registered in reports/decisions.md four arms trained from the
# pretrained llm.pt with step-level source mixing (S = ext_short, L = long; patterns S / S,S,S,L / S,L /
# S,L,L,L), per-source caps = the frozen E3/E2 caps (L 22500 / S 20700), config = E3 otherwise, cv = dev_long.
# ONE config serves all four arms (the pattern is a --mix_pattern override). Same structural rule as the E4
# stage configs: an E6 header terminated by E6_HEADER_END, then cv3_long_sft.yaml VERBATIM plus the
# `mix_sources` block whose every line carries E6_TAG. The default train.py path must stay untouched: the
# mixing branch is guarded by --mix_pattern (default None) and only that branch imports the mixing module.
# ---------------------------------------------------------------------------------------------------
MIX_CFG = 'cv3_mix_sft.yaml'
E6_TAG = 'E6-MIX'
E6_HEADER_END = '# ==== END E6 MIX HEADER'
E6_DECISION_DATE = '2026-08-30'
# Pre-registered per-source caps: the frozen long / short caps. Changing one needs a new decisions.md line.
E6_CAPS = {'L': FROZEN_CAP[LONG_CFG], 'S': FROZEN_CAP[SHORT_CFG]}
# Lead amendments 2026-08-30 (reports/decisions.md). Amendment 1 gave S a per-source VRAM guard of 34000 (the
# first 149 M31 steps: S 17 363 tokens/step, -7.8 % vs E3); measured on CPU it was insufficient (S 17 415) and is
# SUPERSEDED by amendment 2: the S source gets sort_size 3000 (a ~110-clip ext_short batch straddled a 500-row
# sort chunk every 4-5 batches and closed at 4-9k tokens; with 3000 S averages 20.0k under the frozen guard).
# Hence: NO per-source max_padded_positions in either source (both inherit the yaml 28672, as the cv pass does),
# S sort_size 3000, L no sort_size (yaml 500).
E6_GUARD = {}
E6_GUARD_DEFAULT = FROZEN_BOTH['max_padded_positions']
E6_SORT = {'S': '3000'}
E6_PATTERNS = {'S': 'S', 'M31': 'S,S,S,L', 'M11': 'S,L', 'M13': 'S,L,L,L'}
MIXED_PY = os.path.join(ROOT, 'src', 'training', 'mixed_dataset.py')
EXECUTOR_PY = os.path.join(ROOT, 'src', 'training', 'cosyvoice_train', 'executor.py')
TRAIN_MIX = os.path.join(ROOT, 'scripts', 'train_mix.sh')
RUN_MIX = os.path.join(ROOT, 'scripts', 'run_mix.sh')


def _split_mix():
    lines = _read(MIX_CFG)
    idx = [i for i, l in enumerate(lines) if l.startswith(E6_HEADER_END)]
    assert len(idx) == 1, '{}: expected exactly one {!r} line, found {}'.format(MIX_CFG, E6_HEADER_END, len(idx))
    header, rest = lines[:idx[0] + 1], lines[idx[0] + 1:]
    return header, [l for l in rest if E6_TAG not in l], [l for l in rest if E6_TAG in l]


def test_long_config_carries_no_mix_marker():
    text = '\n'.join(_read(LONG_CFG))
    assert E6_TAG not in text and E6_HEADER_END not in text and 'mix_sources' not in text


def test_mix_config_is_the_long_config_plus_only_the_marked_block():
    header, body, marked = _split_mix()
    assert marked, '{}: no {} lines, so there are no per-source caps'.format(MIX_CFG, E6_TAG)
    long_lines = _read(LONG_CFG)
    # the marked block is followed by one blank separator line, which the verbatim rule tolerates
    body = [l for l in body]
    if len(body) == len(long_lines) + 1:
        blank = [i for i, l in enumerate(body) if l == '' and i < len(long_lines) and long_lines[i] != '']
        assert blank, '{}: one extra line that is not the blank separator after the marked block'.format(MIX_CFG)
        del body[blank[0]]
    assert len(body) == len(long_lines), (
        '{}: below the header and outside the {} lines it must be cv3_long_sft.yaml verbatim, but it has '
        '{} lines against {}'.format(MIX_CFG, E6_TAG, len(body), len(long_lines)))
    diff = [(i + 1, a, b) for i, (a, b) in enumerate(zip(body, long_lines)) if a != b]
    assert not diff, '{}: differs from cv3_long_sft.yaml outside the header and the {} lines: {}'.format(MIX_CFG, E6_TAG, diff[:5])


def test_mix_config_marked_lines_are_only_the_mix_sources_block():
    _, _, marked = _split_mix()
    assert sum(1 for l in marked if l.startswith('mix_sources:')) == 1
    for line in marked:
        code = line.split('#')[0].rstrip()
        assert code == '' or code.startswith('mix_sources:') or code.startswith('    '), \
            '{}: a marked line outside the mix_sources block: {!r}'.format(MIX_CFG, line)
        assert code == '' or any(k in code for k in ('mix_sources', 'L:', 'S:', CAP_KEY, 'max_padded_positions', 'sort_size', 'data_list')), line
    keys = [l.split('#')[0].strip() for l in marked if re.match(r'^    [A-Za-z]+:\s*(#.*)?$', l)]
    assert keys == ['L:', 'S:'], 'exactly the two pre-registered sources, L then S: {}'.format(keys)


@pytest.mark.parametrize('name,cap', sorted(E6_CAPS.items()))
def test_mix_config_per_source_caps_are_the_frozen_arm_caps(name, cap):
    _, _, marked = _split_mix()
    block, inside = [], False
    for l in marked:
        code = l.split('#')[0].rstrip()
        if re.match(r'^    [A-Za-z]+:$', code):
            inside = code.strip() == name + ':'
            continue
        if inside:
            block.append(code)
    hits = [m.group(1) for m in (re.match(r'^\s*' + CAP_KEY + r':\s*(\S+)\s*$', c) for c in block) if m]
    assert hits == [cap], '{}: mix_sources.{}.{} must be exactly {} (frozen), found {}'.format(MIX_CFG, name, CAP_KEY, cap, hits)


def _mix_source_block(name):
    _, _, marked = _split_mix()
    block, inside = [], False
    for l in marked:
        code = l.split('#')[0].rstrip()
        if re.match(r'^    [A-Za-z]+:$', code):
            inside = code.strip() == name + ':'
            continue
        if inside:
            block.append(code)
    return block


@pytest.mark.parametrize('name', sorted(E6_CAPS))
def test_mix_config_per_source_padded_positions_guard(name):
    hits = [m.group(1) for m in (re.match(r'^\s*max_padded_positions:\s*(\S+)\s*$', c) for c in _mix_source_block(name)) if m]
    if name in E6_GUARD:
        assert hits == [E6_GUARD[name]], '{}: mix_sources.{}.max_padded_positions must be exactly {}, found {}'.format(
            MIX_CFG, name, E6_GUARD[name], hits)
    else:
        assert hits == [], '{}: mix_sources.{} must NOT override max_padded_positions (inherits {}), found {}'.format(
            MIX_CFG, name, E6_GUARD_DEFAULT, hits)


@pytest.mark.parametrize('name', sorted(E6_CAPS))
def test_mix_config_per_source_sort_size(name):
    hits = [m.group(1) for m in (re.match(r'^\s*sort_size:\s*(\S+)\s*$', c) for c in _mix_source_block(name)) if m]
    if name in E6_SORT:
        assert hits == [E6_SORT[name]], '{}: mix_sources.{}.sort_size must be exactly {}, found {}'.format(MIX_CFG, name, E6_SORT[name], hits)
    else:
        assert hits == [], '{}: mix_sources.{} must NOT override sort_size (yaml 500), found {}'.format(MIX_CFG, name, hits)


def test_mix_config_header_cites_the_amendments():
    header, _, _ = _split_mix()
    text = '\n'.join(header)
    assert 'amendment 2' in text and 'sort_size: 3000' in text and E6_GUARD_DEFAULT in text
    assert '34000' in text and 'SUPERSEDED' in text, 'amendment 1 must be recorded as superseded, not erased'


def test_mix_config_keeps_the_frozen_long_values_for_the_cv_pass():
    """The unmodified data_pipeline (cv pass, and any non-mixed run of this config) keeps the long cap."""
    body = '\n'.join(_split_mix()[1])
    assert 'max_speech_tokens_in_batch: {}'.format(FROZEN_CAP[LONG_CFG]) in body
    assert 'max_padded_positions: {}'.format(FROZEN_BOTH['max_padded_positions']) in body
    assert 'mix_ratio: {}'.format(FROZEN_BOTH['mix_ratio']) in body
    assert 'warmup_steps: 50' in body and 'save_per_step: 200' in body and 'lr: 1e-5' in body


def test_mix_config_header_cites_the_preregistration():
    header, _, _ = _split_mix()
    for line in header:
        assert not line.strip() or line.startswith('#'), '{}: non-comment line in the header: {!r}'.format(MIX_CFG, line)
    text = '\n'.join(header)
    assert E6_DECISION_DATE in text and 'decisions.md' in text and 'E6' in text and 'Mixed-SFT' in text
    for value in ('S,S,S,L', 'S,L,L,L', 'S,L', '22500', '20700', '28672', '3001', 'dev_long', 'num_workers=2'):
        assert value in text, '{}: header does not document {!r}'.format(MIX_CFG, value)
    assert "mode='dev'" in text and 'dev loss' in text


def test_train_py_default_path_is_untouched_by_mixing():
    src = _text(TRAIN_PY)
    assert "'--mix_pattern', default=None" in src, '--mix_pattern must default to None (off)'
    assert "'--mix_source', action='append', default=None" in src
    assert 'if args.mix_pattern is None:' in src
    branch = src.split('if args.mix_pattern is None:')[1]
    default_branch = branch.split('else:')[0]
    assert 'init_dataset_and_dataloader(args, configs, gan, args.dpo)' in default_branch, \
        'the default branch must build the datasets exactly as before'
    mixed_branch = branch.split('else:')[1][:800]
    assert 'from src.training.mixed_dataset import init_mixed_dataset_and_dataloader' in mixed_branch, \
        'the mixing module may only be imported inside the --mix_pattern branch'
    assert src.count('mixed_dataset import') == 1
    assert 'mix_info = None' in src and "run_info['mix']" in src


def test_executor_logs_the_source_and_keeps_the_old_fields():
    src = _text(EXECUTOR_PY)
    assert "'source': source, 'source_epoch': source_epoch" in src
    for k in ('step', 'epoch', 'speech_tokens', 'text_tokens', 'padded_positions', 'samples', 'step_time_sec',
              'loss', 'acc', 'lr', 'grad_norm', 'peak_vram_alloc_gb', 'peak_vram_reserved_gb', 'rss_gb', 'total_speech_tokens'):
        assert "'{}':".format(k) in src, 'train_stats.jsonl lost the field {}'.format(k)
    assert "batch_dict.get('source')" in src, 'a non-mixed batch has no source key -> None, never a KeyError'


def test_mixed_dataset_module_contract():
    src = _text(MIXED_PY)
    assert 'class MixedDataset(IterableDataset)' in src
    assert 'def set_epoch(self, epoch)' in src
    assert 'raise MixSourceError' in src and 'restarting with epoch' in src
    assert "batch['source'] = name" in src and "batch['source_epoch'] = epoch[name]" in src
    assert 'S,S,S,S,S,S,L,L' in src, 'the worker interleaving must be documented in the module'
    assert "POS_KEY = 'max_padded_positions'" in src and 'def with_token_cap(data_pipeline, cap, max_padded_positions=None, sort_size=None)' in src


def test_train_mix_launcher_wiring():
    src = _text(TRAIN_MIX)
    assert 'CUDA_VISIBLE_DEVICES=1' in src and 'expandable_segments:True' in src
    assert '--mix_pattern "$pattern"' in src and '--mix_source "S=$short_list"' in src and '--mix_source "L=$long_list"' in src
    assert 'configs/train/cv3_mix_sft.yaml' in src
    assert 'data/train/ext_short/parquet/data.list' in src and 'data/train/dev_long/parquet/data.list' in src
    assert '--num_workers 2' in src and '--prefetch 2' in src
    assert 'exp_dir=$WORKDIR/exp/mix' in src
    assert 'src/training/cosyvoice_train/train.py' in src


def test_run_mix_runner_implements_the_preregistration():
    src = _text(RUN_MIX)
    for arm, pat in E6_PATTERNS.items():
        assert '[{}]="{}"'.format(arm, pat) in src, 'pattern map lost {} -> {}'.format(arm, pat)
    assert 'STEPS=${STEPS:-3000}' in src, '3000 optimizer steps -> final checkpoint step 3001'
    assert '_step_$((STEPS+1))' in src, 'only the FINAL checkpoint is generated/evaluated'
    assert '--keep_best 1' in src
    assert 'check_budget_mix.py' in src and '--ref exp/long/pilot' in src and 'budget_full.json' in src
    assert '--mode native' in src and '--seed 0' in src and "--voice-filter 'ref_(female|male)_01$'" in src
    assert '--floor-cache results/$BASE/asr_floor.jsonl' in src
    assert 'E6_${arm}_${cn}' in src
    for label in ('E1_native=results/$BASE/E1', 'E2_short=results/$SFT/E2_epoch_3_step_3001',
                  'E3_long=results/$SFT/E3_epoch_3_step_3001', 'E3_long_773=results/$SFT/E3_epoch_0_whole',
                  'E4_curr=results/$CURR/E4_epoch_0_whole'):
        assert label in src, 'comparator row missing: {}'.format(label)
    assert 'TRAINED' in src, 'idempotency marker'
    assert 'ABORT: missing $SHORT_LIST' in src, 'must refuse to start without the ext_short list'
    assert 'scripts/train_arm.sh' not in src.split('set -uo pipefail')[1], 'the runner must use train_mix.sh, not train_arm.sh'



# ---------------------------------------------------------------------------------------------------
# E7 Punct-SFT (A6-mix, 2026-08-30). Pre-registered by the Lead: long-arm SFT on the manifest's text_e2e
# (punctuated) instead of ROVER; NOT ONE training value changes. The config is therefore an E7 header block
# terminated by E7_HEADER_END followed by cv3_long_sft.yaml VERBATIM (zero inserted lines): only the CLI
# data lists differ (train_arm.sh --train_list/--cv_list, scripts/run_punct.sh).
# ---------------------------------------------------------------------------------------------------
PUNCT_CFG = 'cv3_punct_sft.yaml'
E7_HEADER_END = '# ==== END E7 PUNCT HEADER'
E7_DECISION_DATE = '2026-08-30'


def _split_punct():
    lines = _read(PUNCT_CFG)
    idx = [i for i, l in enumerate(lines) if l.startswith(E7_HEADER_END)]
    assert len(idx) == 1, '{}: expected exactly one {!r} line, found {}'.format(PUNCT_CFG, E7_HEADER_END, len(idx))
    return lines[:idx[0] + 1], lines[idx[0] + 1:]


def test_punct_config_is_the_long_config_verbatim():
    header, body = _split_punct()
    for line in header:
        assert not line.strip() or line.startswith('#'), '{}: non-comment line in the header: {!r}'.format(PUNCT_CFG, line)
    long_lines = _read(LONG_CFG)
    assert len(body) == len(long_lines), '{}: body must be cv3_long_sft.yaml verbatim ({} lines vs {})'.format(
        PUNCT_CFG, len(body), len(long_lines))
    diff = [(i + 1, a, b) for i, (a, b) in enumerate(zip(body, long_lines)) if a != b]
    assert not diff, '{}: differs from cv3_long_sft.yaml below the header: {}'.format(PUNCT_CFG, diff[:5])


def test_punct_config_header_cites_the_preregistration():
    header, _ = _split_punct()
    text = '\n'.join(header)
    assert E7_DECISION_DATE in text and 'decisions.md' in text and 'E7' in text and 'Punct' in text
    for value in ('text_e2e', 'long_punct', 'dev_long_punct', '--train_list', '--cv_list', 'RULONG_GPU',
                  '0.252', '0.298', '0.014', 'VERBATIM'):
        assert value in text, '{}: header does not document {!r}'.format(PUNCT_CFG, value)

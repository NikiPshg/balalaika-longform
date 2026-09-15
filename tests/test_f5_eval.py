"""F5 uses the existing content decision without inventing EOS events."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('f5_evaluate', ROOT/'scripts/f5_evaluate.py')
f5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f5)


class FakeASR:
    model_id = 'gigaam-v3-rnnt'

    def vad_fingerprint(self):
        return {'vad_options_hash': 'test'}

    def transcribe(self, path):
        text = Path(path).read_text()
        return SimpleNamespace(text=text, error=None, segments=[], model_id=self.model_id,
                               onnx_asr_version='test', provider='CPU', elapsed_sec=0,
                               speed_x_realtime=1, audio_duration_sec=10)


def test_flow_coverage_and_missing_audio_stay_in_denominator(tmp_path):
    pytest.importorskip('jiwer')  # ASR/content tests run in .venv-eval, not the F5 environment.
    text = 'мама мыла окно потом папа принёс большую красивую зелёную корзину'
    benchmark = {'t': dict(text_id='t', text_ref=text, bucket='B0',
                           human_audio_path=None, human_duration_sec=None,
                           human_offset_start=None, human_offset_end=None)}
    rows = []
    for name, hyp, status in [('full', text, 'complete'),
                              ('tail_missing', 'мама мыла окно', 'complete'),
                              ('empty', '', 'complete'), ('oom', None, 'oom')]:
        path = tmp_path/f'{name}.wav'
        if hyp is not None:
            path.write_text(hyp)
        rows.append(dict(run_id=name, text_id='t', experiment_id='F5',
                         output_path=str(path), gen_status=status,
                         stop_reason='flow_completed' if status=='complete' else 'oom'))
    scored = f5.evaluate_flow(rows, benchmark, FakeASR(), {}, 'lenient')
    assert [r['status'] for r in scored] == ['complete', 'incomplete_text',
                                            'empty_or_invalid_audio', 'oom']
    assert len(scored) == 4
    assert sum(r['status_complete'] for r in scored) == 1
    assert scored[-1]['wer'] == 1
    assert all('eos' not in r['final_status_reason'] for r in scored)


def test_only_stop_semantics_change():
    original = f5.shared.load_eval_config()
    modified = f5.flow_config()
    modified['final_status']['require_stop_reason_for_complete'] = 'eos'
    assert modified == original


@pytest.mark.parametrize('bad_reason', ['eos', 'duration_reached', None])
def test_does_not_accept_fake_eos_or_ambiguous_stop(bad_reason):
    row = dict(run_id='r', text_id='t', experiment_id='F5', output_path='',
               gen_status='complete', stop_reason=bad_reason)
    with pytest.raises(ValueError, match='never EOS'):
        f5.validate_flow_runs([row], {'t': {}})

"""Concurrent generation resumes from the locked grid without allocating a model."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

pytest.importorskip('torch')  # Run in the isolated F5 environment.
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('f5_generate', ROOT/'scripts/f5_generate.py')
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)


@pytest.fixture
def job(tmp_path, monkeypatch):
    checkpoint, vocab = tmp_path/'checkpoint.pt', tmp_path/'vocab.txt'
    checkpoint.write_bytes(b'test checkpoint; never loaded')
    vocab.write_text('test vocab')
    vocoder = tmp_path/'vocoder'
    vocoder.mkdir()
    (vocoder/'pytorch_model.bin').write_bytes(b'test vocoder; never loaded')
    benchmark, references = tmp_path/'benchmark.jsonl', tmp_path/'references.jsonl'
    benchmark.write_text(json.dumps(dict(text_id='text', text_tts='Русская речь.', bucket='B0'))+'\n')
    references.write_text(json.dumps(dict(voice_id='voice', role='primary', ref_text='Образец.',
                                          wav_path=str(tmp_path/'reference.wav')))+'\n')
    out = tmp_path/'output'
    argv = ['f5_generate.py', '--checkpoint', str(checkpoint), '--vocab', str(vocab),
            '--vocoder-dir', str(vocoder), '--benchmark', str(benchmark),
            '--references', str(references), '--output-dir', str(out),
            '--experiment', 'F5-test', '--mode', 'chunked']
    monkeypatch.setattr(sys, 'argv', argv)

    class FakeGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, **kwargs):
            # Failed attempts also remain recorded; duplicate invocations must
            # not silently retry them and improve the benchmark denominator.
            return dict(gen_status='oom', stop_reason='cuda_oom',
                        output_path=kwargs['output_path'], elapsed=0.1)

    monkeypatch.setattr(generate, 'F5Generator', FakeGenerator)
    generate.main()
    return out, argv


def reject_model_initialization(*args, **kwargs):
    raise AssertionError('Completed grids must not initialize a model or CUDA')


def test_completed_exact_grid_is_noop_before_model_load(job, monkeypatch):
    out, _ = job
    before = (out/'runs.jsonl').read_bytes()
    monkeypatch.setattr(generate, 'F5Generator', reject_model_initialization)
    # Recover the final progress record too if a previous process exited after
    # writing its last run but before writing progress.json.
    (out/'progress.json').unlink()
    generate.main()
    assert (out/'runs.jsonl').read_bytes() == before
    assert json.loads((out/'progress.json').read_text())['completed_grid'] is True


def test_completed_grid_still_validates_protocol(job, monkeypatch):
    _, argv = job
    argv = [*argv]
    argv[-1] = 'uninterrupted'
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(generate, 'F5Generator', reject_model_initialization)
    with pytest.raises(ValueError, match='protocol differs'):
        generate.main()


def test_recorded_runs_without_protocol_are_not_certified(job, monkeypatch):
    out, _ = job
    (out/'protocol.json').unlink()
    monkeypatch.setattr(generate, 'F5Generator', reject_model_initialization)
    with pytest.raises(ValueError, match='original generation protocol'):
        generate.main()
    assert not (out/'protocol.json').exists()


def test_resume_rejects_extra_recorded_runs(job, monkeypatch):
    out, _ = job
    with (out/'runs.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(run_id='unrequested-run'))+'\n')
    monkeypatch.setattr(generate, 'F5Generator', reject_model_initialization)
    with pytest.raises(ValueError, match='outside the requested exact grid'):
        generate.main()


def test_waiting_process_rereads_runs_after_lock_release(job, tmp_path):
    out, argv = job
    saved_runs = (out/'runs.jsonl').read_bytes()
    (out/'runs.jsonl').unlink()
    (out/'progress.json').unlink()
    ready = tmp_path/'waiting-for-lock'
    child_source = textwrap.dedent('''
        import importlib.util, json, sys
        from contextlib import contextmanager
        from pathlib import Path
        module_path, ready_path, argv_json = sys.argv[1:]
        spec = importlib.util.spec_from_file_location('f5_generate', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_lock = module.output_directory_lock
        @contextmanager
        def announced_lock(directory):
            Path(ready_path).write_text('waiting')
            with original_lock(directory):
                yield
        def forbidden_model(*args, **kwargs):
            raise AssertionError('Waiter read stale runs or initialized a model')
        module.output_directory_lock = announced_lock
        module.F5Generator = forbidden_model
        sys.argv = json.loads(argv_json)
        module.main()
    ''')
    process = None
    try:
        with generate.output_directory_lock(out):
            process = subprocess.Popen(
                [sys.executable, '-u', '-c', child_source, str(ROOT/'scripts/f5_generate.py'),
                 str(ready), json.dumps(argv)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
            deadline = time.monotonic() + 15
            while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.01)
            assert ready.exists(), 'Child did not reach the directory lock'
            time.sleep(0.05)
            assert process.poll() is None, 'Child failed to wait for the held directory lock'
            # Simulate the owner recording its final attempt while the other
            # process waits. The waiter must read these bytes after acquiring.
            (out/'runs.jsonl').write_bytes(saved_runs)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stdout + stderr
        assert 'no model initialization needed' in stdout
        assert (out/'runs.jsonl').read_bytes() == saved_runs
        assert json.loads((out/'progress.json').read_text())['completed_grid'] is True
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()

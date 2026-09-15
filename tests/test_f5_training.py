"""Meaningful CPU checks for matched exposure, resume ordering, and flow inference."""
import json
from pathlib import Path
import random

import numpy as np
import pytest
torch = pytest.importorskip('torch')  # Training tests run in the separate F5 environment.

from src.f5_training.common import (capture_rng, configure_cuda, create_ema, ema_to_model_state, estimated_duration_frames,
                                    load_audio, parent_groups, read_manifest, restore_rng)
from src.f5_training.infer import chunk_text, crossfade
from src.f5_training.train import learning_rate, parent_at_update, save_checkpoint


@pytest.mark.parametrize("visible", ["0", "1"])
def test_cuda_visibility_maps_one_authorized_gpu_to_logical_zero(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    settings = {}
    for backend in ("flash", "mem_efficient", "math"):
        monkeypatch.setattr(torch.backends.cuda, f"enable_{backend}_sdp",
                            lambda enabled, name=backend: settings.update({name: enabled}))
    assert configure_cuda() == torch.device("cuda:0")
    assert settings == dict(flash=True, mem_efficient=True, math=False)


@pytest.mark.parametrize("visible", [None, "", "0,1", "1,0", "2", "-1", " 1", "GPU-unknown"])
def test_cuda_visibility_rejects_unauthorized_or_multiple_gpus_before_cuda_query(monkeypatch, visible):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    def unexpected_cuda_query():
        raise AssertionError("Rejected visibility must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_query)
    with pytest.raises(RuntimeError, match="exactly 0 or 1"):
        configure_cuda()


def test_parent_order_matches_across_views_and_resume():
    long = [{"parent": p, "utt": p} for p in "dcab"]
    short = [{"parent": p, "utt": f"{p}_{n}", "offset_start": n} for p in "badc" for n in range(3)]
    lp, sp = list(parent_groups(long)), list(parent_groups(short))
    assert [parent_at_update(lp, i, 5) for i in range(11)] == [parent_at_update(sp, i, 5) for i in range(11)]
    first = [parent_at_update(lp, i, 5) for i in range(7)]
    resumed = [parent_at_update(lp, i, 5) for i in range(7, 11)]
    assert first + resumed == [parent_at_update(lp, i, 5) for i in range(11)]
    assert len({parent_at_update(lp, i, 5)[0] for i in range(4)}) == 4


def test_frame_weighted_short_gradient_equals_full_frame_gradient():
    # Equal-window weighting would overweight the short tail. Actual-frame
    # weighting must give the same gradient as a concatenated frame mean.
    weight = torch.tensor(0.3, requires_grad=True)
    windows = [torch.tensor([1., 2., 3., 4.]), torch.tensor([9.])]
    count = sum(len(window) for window in windows)
    for window in windows:
        ((weight * window).square().mean() * len(window) / count).backward()
    accumulated = weight.grad.clone()
    weight.grad = None
    (weight * torch.cat(windows)).square().mean().backward()
    torch.testing.assert_close(accumulated, weight.grad)


def test_rng_snapshot_restores_corruption_stream():
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.randn(4))
    restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.randn(4))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_checkpoint_resume_reproduces_next_optimizer_and_ema_update(tmp_path: Path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = create_ema(model)

    def step():
        optimizer.zero_grad()
        x = torch.randn(3, 4) * (random.random() + np.random.rand())
        model(x).square().mean().backward()
        optimizer.step()
        ema.update()

    # Resume just before scheduled EMA update 121, after the initial copy-only
    # warmup, so the test exercises real averaging rather than an unchanged copy.
    for _ in range(120):
        step()
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(checkpoint, model, optimizer, ema=ema, update=120, ledger={"mel_frames": 100},
                    protocol={"seed": 0}, state="running")
    step()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    expected_ema = {key: value.clone() for key, value in ema.state_dict().items()}
    saved = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    optimizer.load_state_dict(saved["optimizer"])
    ema.load_state_dict(saved["ema_model_state_dict"], strict=True)
    restore_rng(saved["rng"])
    step()
    assert saved["update"] == 120 and saved["ledger"]["mel_frames"] == 100
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    for key, value in ema.state_dict().items():
        torch.testing.assert_close(value, expected_ema[key], rtol=0, atol=0)
    exported = ema_to_model_state(ema.state_dict())
    assert set(exported) == set(model.state_dict())
    assert any(not torch.equal(exported[key], expected[key]) for key in exported)


def test_pinned_ema_matches_upstream_default_schedule():
    from ema_pytorch import EMA
    model = torch.nn.Linear(2, 1)
    actual = create_ema(model)
    official = EMA(model, include_online_model=False)
    for step in range(150):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(step / 100)
        actual.update()
        official.update()
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, official.state_dict()[key], rtol=0, atol=0)
    assert actual.step.item() == 150


def test_audio_end_tolerance_is_bounded_and_accounts_actual_samples(tmp_path: Path):
    import soundfile as sf
    audio = tmp_path / "source.wav"
    sf.write(audio, np.ones(24000, dtype=np.float32) * 0.01, 24000)
    row = dict(audio=str(audio), utt="rounded", offset_start=0, offset_end=1.005)
    assert len(load_audio(row)) == 24000
    assert row["_source_samples"] == 24000
    assert row["_actual_audio_seconds"] == 1.0
    assert row["_metadata_end_overrun_samples"] == 120
    with pytest.raises(ValueError, match="Invalid audio offsets"):
        load_audio(dict(audio=str(audio), utt="invalid", offset_start=0, offset_end=1.006))


def test_duration_uses_reference_only_and_utf8_lengths():
    prompt_samples, reference, target = 24000 * 8, "Это образец. ", "Это длинный текст для синтеза."
    frames = prompt_samples // 256
    expected = frames + int(frames / len(reference.encode("utf-8")) * len(target.encode("utf-8")))
    assert estimated_duration_frames(prompt_samples, reference, target) == expected
    assert estimated_duration_frames(prompt_samples, reference, target, speed=2) < expected
    with pytest.raises(ValueError):
        estimated_duration_frames(prompt_samples, "", target)


def test_chunking_keeps_oversize_sentence_and_crossfade_duration():
    # Stock F5 does not subdivide an oversized sentence: do not silently invent
    # stronger chunking that improves a benchmark versus the pinned implementation.
    long_sentence = "Очень " * 40 + "длинное предложение."
    chunks = chunk_text("Первая фраза. " + long_sentence, max_chars=50)
    assert chunks == ["Первая фраза.", long_sentence]
    waves = [np.ones(24000, dtype=np.float32), np.zeros(12000, dtype=np.float32)]
    assert len(crossfade(waves, 0.15)) == 36000 - 3600
    assert len(crossfade(waves, 0)) == 36000


def test_manifest_rejects_duplicate_rows(tmp_path: Path):
    row = dict(utt="a", parent="p", audio="unused.wav", duration=1.0, text="Речь")
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest)


def test_learning_rate_resume_uses_absolute_update():
    rates = [learning_rate(i, 100, 1e-5, 0.05) for i in range(100)]
    assert rates[:5] == pytest.approx([2e-6, 4e-6, 6e-6, 8e-6, 1e-5])
    assert rates[70:] == [learning_rate(i, 100, 1e-5, 0.05) for i in range(70, 100)]
    assert 0 < rates[-1] < rates[50] < rates[5]

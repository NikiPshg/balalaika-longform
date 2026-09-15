# Evaluation protocol and entrypoints

## Main comparison

`artifacts/published50/data/benchmark.jsonl` contains 50 text–voice pairs: 14 short (~75 words), 14 medium (~300 words), and 22 long (~1,200 words). The texts span 22 works and the prompts span 21 voices. Eight voices are new external held-out references and 13 are earlier held-out references. All 13 conditions receive the same paired, punctuated, cased text and reference audio. The main comparison uses continuous generation, including F5-TTS.

The 50-pair subset was chosen after the larger evaluation had begun. Its deterministic selection used metadata, not synthesis scores. The protocol and selection audit retain this timing. Do not describe this smaller subset as having been frozen before any generation, or as the separate sealed benchmark. The earlier two-voice, 60-attempt CosyVoice comparison is included solely to reproduce the corresponding earlier table.

The statistical specification identifies the clustering, report groups and paired contrasts. Resampling uses 10,000 draws and seed 20260913. The reproduction script fails if the saved attempts do not form the expected paired comparison. These intervals characterize variability over the sampled evaluation material; they do not measure variation across independent training seeds.

## Metrics

Full-text WER compares the ASR transcript with the complete intended passage. Missing speech contributes deletions, so an early stop can produce very high WER even when a short surviving fragment sounds clear. The main table uses an unweighted mean of per-attempt WER. Failed attempts are retained. ASR uses GigaAM-v3 RNNT with the frozen normalization and VAD settings in the evaluator; the saved transcript and alignment counts are included for every attempt.

DistillMOS and WeSpeaker are evaluated on 5-second windows, with a 2.5-second hop and a final partial window only when it is at least 2.5 seconds. The speaker scorer uses voiced audio within each wall-clock window and excludes windows with insufficient voiced support. The release contains 64,276 MOS windows and 61,346 similarity windows. Eight nearly empty outputs have no eligible windows; their WER attempts remain in the main comparison.

The plot averages eligible clips at each time and shows contributing item counts below. Thin lines are unsmoothed means; the prominent curves use an exponential smoother with coefficient 0.3. Late-time curves describe the surviving clips and should not be interpreted as quality measurements of all originally requested outputs. DistillMOS is an automatic proxy, and cosine similarity is an embedding comparison. Neither is a human listener study.

## Generate and score new outputs

Use separate model environments from [training.md](training.md), restore or download the required base weights, and run from the repository root. Set `CUDA_VISIBLE_DEVICES` yourself; the package does not reserve a particular physical GPU.

Example CosyVoice base invocation, after model setup:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_generation.py \
  --benchmark artifacts/published50/data/benchmark.jsonl \
  --references artifacts/published50/data/references.jsonl \
  --per-item-voices --mode native --experiment cosy_base_replay \
  --config configs/models/cosyvoice3_base.yaml \
  --out outputs/cosy_base_replay --seed 0 \
  --watchdog-max-tokens 30000 --watchdog-max-seconds 2400
```

`--per-item-voices` is essential: each row specifies its own reference, rather than requesting the Cartesian product of all texts and voices. Qwen and VoxCPM use `scripts/run_generation_qwen.py` and `scripts/run_generation_voxcpm.py` with analogous arguments. Use `--weights` for the appropriate fine-tuned checkpoint. F5 uses `scripts/f5_generate.py` with `--mode uninterrupted`, checkpoint, vocabulary and vocoder paths. All CLIs expose `--help`.

For an exact replay, use the per-arm inference settings recorded in the archived generation logs and `artifacts/runtime_configuration.json`, including watchdog overrides. The base YAML files retain the earlier defaults; the expanded evaluation launch settings can override them. Newly generated results belong in a new directory, not in `artifacts/published50/`.

Score a new autoregressive run with the original evaluator:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_evaluation.py \
  --run-manifest outputs/cosy_base_replay/runs.jsonl \
  --benchmark artifacts/published50/data/benchmark.jsonl \
  --output-dir outputs/cosy_base_replay/evaluation --skip-floor
```

Use `scripts/f5_evaluate.py` for its manifest schema. The ASR environment requires `onnx-asr==0.12.0`, a compatible ONNX runtime and the GigaAM model; those are deliberately outside the CPU review requirements. Window scorer entrypoints are `src/eval/distillmos_windows.py` and `src/eval/spk_sim_windows.py`. Recompute these from original or regenerated WAVs.

## Interpretation limits

The corpus and budgeted adaptation do not establish a universal solution to long-form TTS. F5-TTS remains weak on the longest passages. Within-model Short/Long comparisons are the relevant contrasts; the architecture families have different training/inference behavior and are not matched by parameter count or GPU wall time. VoxCPM's context filter excludes 144 long training units, accounting for about 18.2% of long-form seconds. This changes its effective long-form training set. Training-seed replication and a human listening study are not part of this release.

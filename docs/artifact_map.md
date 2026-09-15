# Artifact map and verification scope

| Paper component | Frozen source | Reproduction |
| --- | --- | --- |
| Manuscript PDFs | `paper/main.tex`, bibliography, styles and two figure PDFs | `python paper/build.py` rebuilds both PDF variants |
| Corpus split table | `data/frozen_manifests/{train_long,dev,test}.jsonl.gz` | `scripts/reproduce.py` sums counts, source videos and seconds |
| Two-reference CosyVoice table | `artifacts/two_voice/*.jsonl` | Recomputes mean WER and completion rates for all 60 attempts per arm |
| Main model table and WER figure | `artifacts/published50/results/*/per_item.jsonl` | Recomputes the 650-attempt analysis and compares every published cell and interval |
| Paired intervals | `artifacts/published50/statistics_spec.json` | 10,000 cluster bootstrap resamples, fixed seed |
| Quality and similarity curves | `artifacts/published50/window_metrics/*per_window.parquet` | Regroups windows, draws support counts and plots |
| Training settings | `configs/train/`, `artifacts/training/` | Configuration inspection; full retraining is separate |

`artifacts/published50/generation/` preserves model identifiers, inference settings, stopping reasons and recorded checkpoint metadata. `output_path` in archived logs identifies the original logical run artifact; most such WAVs are not included. Paths under `artifacts/expanded_source/`, `external/`, or historical `results/` in archived provenance refer to non-bundled source artifacts.

Source files were selected from the working experiment tree, not reconstructed from its older Git HEAD. `artifacts/source_provenance.json` records original hashes and hashes after relocation. Local machine paths were replaced, model/device defaults made configurable, and reviewer-facing entrypoints added. No saved numeric observation or ASR transcript was edited to change a result.

The default pytest suite checks metric edge cases, alignment, failure accounting, analytic bootstrap behavior, adapters' CPU contracts and the current report pipeline. Three archive-dependent tests can be skipped when the old ASR-floor and smoke fixtures are absent. Extra test files cover optional training/model integrations and may require their corresponding dependencies and generated inputs.

The package validates analysis from measurements. It does not claim a bit-for-bit repeat of stochastic GPU training, or an independent repeat of the original synthesis/ASR runs. No fine-tuned model checkpoints are included. Runtime versions read from retained environments are recorded in `artifacts/runtime_versions.json`; that inventory is not an assertion that every package remained unchanged since the original training date.

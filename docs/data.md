# Data and frozen training views

The audio release is [lab260/Balalaika-longform](https://huggingface.co/datasets/lab260/Balalaika-longform), pinned for this package at `96a1abc07caf80703e4dc0578aadb4e43811398d`. The [dataset card](../data/DATASET_CARD.md) describes the audio, attribution and WebDataset schema. Each key has `.flac`, `.txt` and `.json` members. The JSON retains the source video URL, segment boundaries and license evidence.

The complete release contains 6,380 recordings and 189.129808 hours. The default splits contain 6,043 training, 120 validation and 193 test recordings. A separate excluded configuration contains 24 recordings; those recordings are not training examples.

## Restore inputs

The following command needs only the Python standard library:

```bash
python scripts/prepare_corpus.py
```

It expands the frozen experiment manifests into `data/manifests/`. The exact training text files and crop boundaries are already in `data/train/`. `wav.scp` paths resolve from the repository root into `data/corpus/audio/`. `utt2offset` is measured in seconds within the parent segment FLAC, not within the original video.

To download and restore the entire corpus, place the repository and cache on a disk with sufficient free space. The tar files total about 25.2 GB; extracted audio needs additional space.

```bash
python -m pip install huggingface_hub
python scripts/prepare_corpus.py --download --cache-dir /path/on/large/disk/balalaika-shards
```

If the shards are already downloaded:

```bash
python scripts/prepare_corpus.py --shards /path/to/Balalaika-longform
```

Audio is restored under the frozen filenames and checked against the original per-segment SHA-256 values. Existing different audio or manifests cause an error. `--limit 1` is available for a small loading check; it does not constitute complete corpus preparation.

## Preserve the historical experiment

The code package freezes the text and exclusions used in the measured runs. The later dataset release repaired two originally missing transcripts. Those repairs do not retroactively alter the frozen training views or reported results. For a comparison with the paper, use the frozen files rather than rebuilding the views from the newer `.txt` members.

The restore script writes minimal sidecars with the original ASR-agreement values for Qwen preparation. It does not reconstruct word timestamps. Punctuated short-window texts are already supplied, so timestamp-based segmentation does not need to be rerun. Scripts that reconstruct the corpus from raw video annotations are retained as source documentation and require those upstream annotations.

The main evaluation manifest and its 21 prompts are independent of these training manifests. Text attribution is in `artifacts/published50/data/benchmark.jsonl`; prompt provenance is in `artifacts/published50/data/references.jsonl`.

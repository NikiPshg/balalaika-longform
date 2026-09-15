# Third-party notices

The root Apache-2.0 license applies to project code. It does not relicense downloaded models, recordings, reference prompts or source literature.

- Qwen source is included under `third_party/Qwen3-TTS/`, with its existing Apache-2.0 license and copyright notices. Its model/training adaptations are preserved in the snapshot identified in `docs/training.md`.
- CosyVoice-derived training source retains its existing notices. The upstream Apache-2.0 license is also included as `third_party/licenses/CosyVoice.txt`.
- VoxCPM-derived training source retains its existing notices. Its upstream license is in `third_party/licenses/VoxCPM.txt`.
- F5-TTS source dependencies retain the license in `third_party/licenses/F5-TTS.txt`. The Russian base model's saved Hub metadata identifies `cc-by-nc-4.0`; this code license does not grant different rights to those model weights.
- The Balalaika audio release preserves the source videos' Creative Commons attribution metadata. The retained source flags do not establish a single license version for every video. Read `data/LICENSE.md`, `data/DATASET_CARD.md` and the per-sample source metadata.
- External reference prompts preserve their RuLS/LibriVox source information, reader attribution and source license statements in `artifacts/published50/data/references.jsonl`. Where the source Hub card and original collection describe terms differently, both statements are retained.
- Evaluation passages preserve title, author and source URL in `artifacts/published50/data/benchmark.jsonl`. Literary texts and reference recordings are not covered by the code's Apache license.

The project depends on other open-source Python packages listed in the requirements and upstream projects; their own licenses apply to those packages.

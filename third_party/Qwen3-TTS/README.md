# Qwen3-TTS training snapshot

This source snapshot contains the Qwen3-TTS model implementation and the SFT
trainer used by the long-form experiments. It is provided so the custom training
path can be inspected without access to a separate working checkout.

Original snapshot revision: `8c5a685be79b39dce931bdef76ef7e88c99d013d`.
Only source files and configuration templates are included. Weights, virtual
environments, caches, logs, credentials, and unrelated experiment outputs are
excluded. Machine-specific paths have been relocated for this release.

The code retains its upstream Apache-2.0 license and attribution. Start with
`lora_finetuning/trl/` for the training entry points. The main repository's
`docs/training.md` describes the experiment configurations and dependencies.

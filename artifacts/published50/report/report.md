# Limited evaluation: complete predefined 50-attempt grid

This report covers the explicit **main13** group: 13 continuous systems and 0 chunked controls. Each included system has all 50 fixed text–voice pairs. The larger 120-pair evaluation is deferred and is not described as complete.

The user changed the evaluation budget to 50 examples with a 12-hour deadline after seeing Base/Short summaries and Long-run progress. This subset was selected after those observations; it is not a pre-TTS selection. The selector used source metadata only; the timing of the scope amendment remains a limitation that the bootstrap does not remove.

Coverage: 22 conservative work clusters, 22 passages and 21 reference voices. Length-specific attempts per system: 75 words: 14; 300 words: 14; 1200 words: 22.

WER and correct-word recall include every planned attempt, including generation or recognition failures. Values above 100% WER are retained. The longest-input table therefore has its actual B4 subset denominator, not 50 independent long passages.

The four Long-minus-Short comparisons remain predefined. Pointwise 95% intervals use the unchanged 10,000-resample source-cluster bootstrap; crossed source/voice intervals are reported separately. Repeated speakers and nested texts are dependent. Intervals condition on checkpoints and inference seeds and exclude ASR systematic and retraining uncertainty.

Omitted from this report: cosy_base_chunked, f5_base_chunked. An omitted arm is not an observed synthesis failure. Its omission is not evidence of model performance.

No floor-dependent completion, human MOS or listener-preference claim is published. Word-length labels do not imply human-recording durations. The manuscript is not modified by this script.

## Primary contrasts at the longest word condition

- CosyVoice3: Long − Short WER -52.7 pp [-62.7, -42.2].
- Qwen3-TTS: Long − Short WER -52.7 pp [-61.7, -43.4].
- VoxCPM2: Long − Short WER -41.5 pp [-47.7, -35.4].
- F5-TTS: Long − Short WER -4.6 pp [-4.9, -4.4].

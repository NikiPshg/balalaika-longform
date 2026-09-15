# Expanded evaluation: paired all-attempt results

Every number includes the full frozen attempt grid. WER and recall are macro averages.

Intervals condition on fitted checkpoints and inference seeds. The primary bootstrap resamples source/book clusters with selected voices fixed; crossed source/voice resampling is reported separately.

## B0

14 attempts per arm; 14 source clusters, 14 root passages, 11 reference voices.

| Arm | WER-all % | Correct-word recall % |
|---|---:|---:|
| cosy_base | 7.28 | 92.89 |
| cosy_short | 7.65 | 92.60 |
| cosy_long | 5.21 | 95.14 |
| cosy_long_punct | 3.57 | 96.60 |
| qwen_base | 3.85 | 96.32 |
| qwen_short | 10.03 | 90.50 |
| qwen_long | 8.29 | 91.97 |
| vox_base | 12.38 | 87.70 |
| vox_short | 4.44 | 95.73 |
| vox_long | 3.40 | 97.04 |
| f5_base | 45.04 | 58.69 |
| f5_short | 44.20 | 61.17 |
| f5_long | 27.37 | 83.29 |

Paired differences A − B, in percentage points. Intervals are pointwise.

| A − B | Metric | Difference | Source CI | Crossed source/voice CI |
|---|---|---:|---:|---:|
| cosy_long − cosy_short | wer_all_pct | -2.44 | [-4.42, -0.38] | [-5.56, 0.97] |
| cosy_long − cosy_short | correct_word_recall_pct | 2.55 | [0.48, 4.48] | [-0.85, 5.60] |
| qwen_long − qwen_short | wer_all_pct | -1.74 | [-4.76, 1.39] | [-6.69, 3.68] |
| qwen_long − qwen_short | correct_word_recall_pct | 1.46 | [-1.57, 4.36] | [-3.74, 6.27] |
| vox_long − vox_short | wer_all_pct | -1.04 | [-2.43, 0.24] | [-3.54, 0.85] |
| vox_long − vox_short | correct_word_recall_pct | 1.32 | [0.26, 2.46] | [-0.37, 3.39] |
| f5_long − f5_short | wer_all_pct | -16.82 | [-22.50, -10.80] | [-26.20, -4.81] |
| f5_long − f5_short | correct_word_recall_pct | 22.11 | [16.18, 27.66] | [11.24, 31.23] |
| cosy_long − cosy_base | wer_all_pct | -2.07 | [-5.09, 1.27] | [-7.06, 4.06] |
| cosy_long − cosy_base | correct_word_recall_pct | 2.25 | [-1.07, 5.29] | [-3.95, 7.29] |
| qwen_long − qwen_base | wer_all_pct | 4.44 | [1.92, 7.16] | [0.12, 9.23] |
| qwen_long − qwen_base | correct_word_recall_pct | -4.35 | [-6.90, -1.98] | [-8.88, -0.32] |
| vox_long − vox_base | wer_all_pct | -8.98 | [-15.04, -3.78] | [-19.42, -1.63] |
| vox_long − vox_base | correct_word_recall_pct | 9.35 | [3.95, 15.53] | [1.68, 20.06] |
| f5_long − f5_base | wer_all_pct | -17.67 | [-23.81, -10.71] | [-27.23, -3.94] |
| f5_long − f5_base | correct_word_recall_pct | 24.60 | [16.84, 31.40] | [9.68, 35.17] |
| cosy_long_punct − cosy_long | wer_all_pct | -1.64 | [-3.87, 0.28] | [-5.97, 1.44] |
| cosy_long_punct − cosy_long | correct_word_recall_pct | 1.45 | [-0.51, 3.86] | [-1.57, 6.22] |
| cosy_long_punct − cosy_base | wer_all_pct | -3.71 | [-6.68, -1.24] | [-9.18, -0.01] |
| cosy_long_punct − cosy_base | correct_word_recall_pct | 3.71 | [1.28, 6.66] | [0.15, 9.15] |

## B2

14 attempts per arm; 14 source clusters, 14 root passages, 11 reference voices.

| Arm | WER-all % | Correct-word recall % |
|---|---:|---:|
| cosy_base | 84.75 | 15.53 |
| cosy_short | 82.23 | 18.67 |
| cosy_long | 13.81 | 90.50 |
| cosy_long_punct | 5.56 | 94.70 |
| qwen_base | 5.30 | 95.13 |
| qwen_short | 49.55 | 50.73 |
| qwen_long | 7.70 | 92.72 |
| vox_base | 14.21 | 86.10 |
| vox_short | 4.06 | 96.26 |
| vox_long | 2.41 | 97.76 |
| f5_base | 98.71 | 1.29 |
| f5_short | 97.59 | 2.44 |
| f5_long | 81.40 | 19.05 |

Paired differences A − B, in percentage points. Intervals are pointwise.

| A − B | Metric | Difference | Source CI | Crossed source/voice CI |
|---|---|---:|---:|---:|
| cosy_long − cosy_short | wer_all_pct | -68.42 | [-79.15, -56.47] | [-85.50, -49.35] |
| cosy_long − cosy_short | correct_word_recall_pct | 71.83 | [62.71, 80.33] | [57.00, 85.91] |
| qwen_long − qwen_short | wer_all_pct | -41.85 | [-53.58, -29.88] | [-59.69, -21.94] |
| qwen_long − qwen_short | correct_word_recall_pct | 41.99 | [29.98, 53.79] | [22.00, 59.93] |
| vox_long − vox_short | wer_all_pct | -1.65 | [-2.28, -1.06] | [-2.82, -0.71] |
| vox_long − vox_short | correct_word_recall_pct | 1.50 | [0.95, 2.11] | [0.64, 2.66] |
| f5_long − f5_short | wer_all_pct | -16.19 | [-18.22, -14.45] | [-20.16, -13.43] |
| f5_long − f5_short | correct_word_recall_pct | 16.61 | [14.90, 18.59] | [13.93, 20.45] |
| cosy_long − cosy_base | wer_all_pct | -70.94 | [-82.63, -55.89] | [-88.17, -45.10] |
| cosy_long − cosy_base | correct_word_recall_pct | 74.96 | [66.28, 83.14] | [61.27, 88.27] |
| qwen_long − qwen_base | wer_all_pct | 2.40 | [-1.17, 5.42] | [-4.68, 7.18] |
| qwen_long − qwen_base | correct_word_recall_pct | -2.40 | [-5.32, 1.18] | [-7.00, 4.74] |
| vox_long − vox_base | wer_all_pct | -11.80 | [-26.87, -1.10] | [-38.75, -0.32] |
| vox_long − vox_base | correct_word_recall_pct | 11.66 | [0.97, 26.76] | [0.33, 38.56] |
| f5_long − f5_base | wer_all_pct | -17.31 | [-19.27, -15.64] | [-21.17, -14.76] |
| f5_long − f5_base | correct_word_recall_pct | 17.76 | [16.05, 19.68] | [15.02, 21.44] |
| cosy_long_punct − cosy_long | wer_all_pct | -8.25 | [-17.11, -1.63] | [-23.61, -0.14] |
| cosy_long_punct − cosy_long | correct_word_recall_pct | 4.20 | [0.45, 9.47] | [-0.33, 14.92] |
| cosy_long_punct − cosy_base | wer_all_pct | -79.19 | [-87.32, -70.27] | [-92.45, -64.68] |
| cosy_long_punct − cosy_base | correct_word_recall_pct | 79.16 | [70.31, 87.19] | [64.68, 92.47] |

## B4

22 attempts per arm; 22 source clusters, 22 root passages, 19 reference voices.

| Arm | WER-all % | Correct-word recall % |
|---|---:|---:|
| cosy_base | 99.80 | 0.20 |
| cosy_short | 99.94 | 0.06 |
| cosy_long | 47.28 | 69.48 |
| cosy_long_punct | 16.64 | 85.94 |
| qwen_base | 66.88 | 34.43 |
| qwen_short | 87.96 | 12.18 |
| qwen_long | 35.24 | 67.07 |
| vox_base | 94.93 | 5.22 |
| vox_short | 90.52 | 10.53 |
| vox_long | 48.98 | 54.87 |
| f5_base | 99.73 | 0.27 |
| f5_short | 99.34 | 0.66 |
| f5_long | 94.70 | 5.36 |

Paired differences A − B, in percentage points. Intervals are pointwise.

| A − B | Metric | Difference | Source CI | Crossed source/voice CI |
|---|---|---:|---:|---:|
| cosy_long − cosy_short | wer_all_pct | -52.67 | [-62.69, -42.24] | [-67.71, -34.63] |
| cosy_long − cosy_short | correct_word_recall_pct | 69.42 | [57.53, 80.02] | [48.47, 85.58] |
| qwen_long − qwen_short | wer_all_pct | -52.72 | [-61.66, -43.39] | [-67.29, -35.72] |
| qwen_long − qwen_short | correct_word_recall_pct | 54.89 | [45.24, 63.89] | [37.15, 69.08] |
| vox_long − vox_short | wer_all_pct | -41.55 | [-47.72, -35.43] | [-51.45, -31.29] |
| vox_long − vox_short | correct_word_recall_pct | 44.34 | [38.34, 50.36] | [34.04, 53.96] |
| f5_long − f5_short | wer_all_pct | -4.64 | [-4.95, -4.35] | [-5.18, -4.17] |
| f5_long − f5_short | correct_word_recall_pct | 4.69 | [4.40, 5.01] | [4.22, 5.27] |
| cosy_long − cosy_base | wer_all_pct | -52.52 | [-62.53, -42.11] | [-67.47, -34.53] |
| cosy_long − cosy_base | correct_word_recall_pct | 69.28 | [57.40, 79.87] | [48.31, 85.39] |
| qwen_long − qwen_base | wer_all_pct | -31.64 | [-39.47, -23.77] | [-45.53, -18.27] |
| qwen_long − qwen_base | correct_word_recall_pct | 32.63 | [24.42, 40.70] | [18.35, 46.59] |
| vox_long − vox_base | wer_all_pct | -45.96 | [-52.89, -39.19] | [-57.22, -35.02] |
| vox_long − vox_base | correct_word_recall_pct | 49.64 | [42.99, 56.41] | [38.59, 60.17] |
| f5_long − f5_base | wer_all_pct | -5.04 | [-5.33, -4.78] | [-5.57, -4.63] |
| f5_long − f5_base | correct_word_recall_pct | 5.09 | [4.81, 5.40] | [4.66, 5.66] |
| cosy_long_punct − cosy_long | wer_all_pct | -30.64 | [-41.12, -20.45] | [-49.31, -14.70] |
| cosy_long_punct − cosy_long | correct_word_recall_pct | 16.46 | [6.77, 27.50] | [2.16, 36.80] |
| cosy_long_punct − cosy_base | wer_all_pct | -83.16 | [-89.27, -76.58] | [-92.90, -72.00] |
| cosy_long_punct − cosy_base | correct_word_recall_pct | 85.73 | [79.48, 91.54] | [74.54, 94.53] |

## ALL

50 attempts per arm; 22 source clusters, 22 root passages, 21 reference voices.

| Arm | WER-all % | Correct-word recall % |
|---|---:|---:|
| cosy_base | 69.68 | 30.45 |
| cosy_short | 69.14 | 31.18 |
| cosy_long | 26.13 | 82.55 |
| cosy_long_punct | 9.88 | 91.37 |
| qwen_base | 31.99 | 68.75 |
| qwen_short | 55.38 | 44.91 |
| qwen_long | 19.98 | 81.22 |
| vox_base | 49.22 | 50.96 |
| vox_short | 42.21 | 58.39 |
| vox_long | 23.18 | 78.69 |
| f5_base | 84.13 | 16.91 |
| f5_short | 83.41 | 18.10 |
| f5_long | 72.12 | 31.01 |

Paired differences A − B, in percentage points. Intervals are pointwise.

| A − B | Metric | Difference | Source CI | Crossed source/voice CI |
|---|---|---:|---:|---:|
| cosy_long − cosy_short | wer_all_pct | -43.01 | [-50.06, -36.56] | [-57.03, -29.20] |
| cosy_long − cosy_short | correct_word_recall_pct | 51.37 | [45.11, 58.53] | [35.84, 66.60] |
| qwen_long − qwen_short | wer_all_pct | -35.40 | [-42.03, -29.06] | [-46.09, -23.16] |
| qwen_long − qwen_short | correct_word_recall_pct | 36.32 | [29.83, 43.13] | [23.71, 47.27] |
| vox_long − vox_short | wer_all_pct | -19.03 | [-23.58, -15.41] | [-26.94, -12.55] |
| vox_long − vox_short | correct_word_recall_pct | 20.30 | [16.95, 24.49] | [13.83, 27.61] |
| f5_long − f5_short | wer_all_pct | -11.28 | [-13.18, -9.19] | [-14.82, -7.55] |
| f5_long − f5_short | correct_word_recall_pct | 12.91 | [10.83, 14.82] | [9.35, 16.70] |
| cosy_long − cosy_base | wer_all_pct | -43.55 | [-51.00, -36.45] | [-58.43, -29.11] |
| cosy_long − cosy_base | correct_word_recall_pct | 52.10 | [45.34, 59.46] | [36.62, 67.88] |
| qwen_long − qwen_base | wer_all_pct | -12.01 | [-15.91, -8.71] | [-19.56, -5.27] |
| qwen_long − qwen_base | correct_word_recall_pct | 12.47 | [8.95, 16.59] | [5.44, 20.41] |
| vox_long − vox_base | wer_all_pct | -26.04 | [-32.93, -20.66] | [-37.84, -17.34] |
| vox_long − vox_base | correct_word_recall_pct | 27.72 | [22.52, 34.35] | [18.97, 39.00] |
| f5_long − f5_base | wer_all_pct | -12.01 | [-14.19, -9.49] | [-15.70, -7.89] |
| f5_long − f5_base | correct_word_recall_pct | 14.10 | [11.29, 16.57] | [9.54, 18.54] |
| cosy_long_punct − cosy_long | wer_all_pct | -16.25 | [-21.07, -11.44] | [-25.44, -7.95] |
| cosy_long_punct − cosy_long | correct_word_recall_pct | 8.82 | [4.37, 13.78] | [1.92, 18.13] |
| cosy_long_punct − cosy_base | wer_all_pct | -59.80 | [-66.04, -54.35] | [-74.58, -45.28] |
| cosy_long_punct − cosy_base | correct_word_recall_pct | 60.93 | [55.23, 67.37] | [46.10, 75.60] |

## B4::new_external_heldout

11 attempts per arm; 11 source clusters, 11 root passages, 8 reference voices.

| Arm | WER-all % | Correct-word recall % |
|---|---:|---:|
| cosy_base | 99.79 | 0.21 |
| cosy_short | 99.92 | 0.08 |
| cosy_long | 43.86 | 72.16 |
| cosy_long_punct | 22.06 | 78.65 |
| qwen_base | 64.15 | 37.99 |
| qwen_short | 88.14 | 11.99 |
| qwen_long | 26.84 | 76.60 |
| vox_base | 95.22 | 5.08 |
| vox_short | 88.24 | 13.41 |
| vox_long | 44.35 | 60.46 |
| f5_base | 99.72 | 0.28 |
| f5_short | 99.42 | 0.58 |
| f5_long | 94.81 | 5.22 |

Paired differences A − B, in percentage points. Intervals are pointwise.

| A − B | Metric | Difference | Source CI | Crossed source/voice CI |
|---|---|---:|---:|---:|
| cosy_long − cosy_short | wer_all_pct | -56.06 | [-68.64, -41.67] | [-73.25, -31.15] |
| cosy_long − cosy_short | correct_word_recall_pct | 72.08 | [60.70, 82.88] | [50.84, 88.71] |
| qwen_long − qwen_short | wer_all_pct | -61.30 | [-70.44, -52.28] | [-76.53, -45.33] |
| qwen_long − qwen_short | correct_word_recall_pct | 64.62 | [56.33, 72.93] | [49.62, 78.55] |
| vox_long − vox_short | wer_all_pct | -43.89 | [-52.72, -34.62] | [-57.42, -27.83] |
| vox_long − vox_short | correct_word_recall_pct | 47.05 | [38.71, 54.11] | [32.23, 57.36] |
| f5_long − f5_short | wer_all_pct | -4.61 | [-5.02, -4.25] | [-5.37, -4.02] |
| f5_long − f5_short | correct_word_recall_pct | 4.64 | [4.27, 5.06] | [4.05, 5.44] |
| cosy_long − cosy_base | wer_all_pct | -55.93 | [-68.56, -41.56] | [-73.18, -30.91] |
| cosy_long − cosy_base | correct_word_recall_pct | 71.94 | [60.53, 82.80] | [50.78, 88.59] |
| qwen_long − qwen_base | wer_all_pct | -37.31 | [-47.68, -27.46] | [-56.05, -21.19] |
| qwen_long − qwen_base | correct_word_recall_pct | 38.62 | [28.47, 48.98] | [21.09, 56.75] |
| vox_long − vox_base | wer_all_pct | -50.87 | [-61.80, -39.61] | [-67.04, -32.47] |
| vox_long − vox_base | correct_word_recall_pct | 55.39 | [45.29, 64.62] | [38.56, 68.27] |
| f5_long − f5_base | wer_all_pct | -4.92 | [-5.31, -4.57] | [-5.63, -4.35] |
| f5_long − f5_base | correct_word_recall_pct | 4.95 | [4.59, 5.35] | [4.36, 5.68] |
| cosy_long_punct − cosy_long | wer_all_pct | -21.80 | [-36.95, -8.00] | [-49.41, -0.84] |
| cosy_long_punct − cosy_long | correct_word_recall_pct | 6.50 | [-3.37, 17.43] | [-8.50, 25.96] |
| cosy_long_punct − cosy_base | wer_all_pct | -77.72 | [-87.76, -67.52] | [-94.08, -60.69] |
| cosy_long_punct − cosy_base | correct_word_recall_pct | 78.44 | [68.42, 88.17] | [61.44, 94.37] |

## Interpretation limits

- Nested lengths and repeated voices are dependent; attempt count is not an independent sample size.
- Sampling intervals condition on the chosen checkpoints and seeds; chosen texts/voices are not a probability sample of all Russian speech.
- Crossed source/voice intervals are a sensitivity analysis; sparse assignments and few levels can make percentile coverage inaccurate.
- Pointwise intervals across multiple backbones, metrics and lengths do not establish familywise significance.

Crossed resampling reference: [Owen, The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111).

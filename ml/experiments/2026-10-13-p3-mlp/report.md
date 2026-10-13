# P3 — `mlp`

**Question:** Does a model learned from HaGRID photos work on live webcam video in the app?

- Train: HaGRID train subjects
- Test: own_v2 (live webcam)
- Date: 2026-10-13
- Reproduce: `uv run python -m ml.eval --protocol p3 --model mlp`

## Datasets

| Dataset | SHA-256 | Records used (raw) | Duplicates removed |
|---|---|---:|---:|
| `hagrid_landmarks_v1` (`ml/data/hagrid_landmarks_v1/records.jsonl`) | `d5db335e537d…` | 12485 (12495) | 10 |
| `own_v2` (`ml/data/own_v2.json`) | `86b72fad77f7…` | 4680 (4680) | 0 |

## Results

4680 test samples, 136 independent groups (bootstrap unit). Trained on 8362 samples from 6753 subjects.

| Metric | Value | 95 % CI |
|---|---:|---|
| accuracy | 0.851 | [0.796, 0.900] |
| macro_f1 | 0.823 | [0.777, 0.857] |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `call_me` | 0.773 | 0.957 | 0.855 | 460 |
| `fist` | 0.909 | 1.000 | 0.952 | 460 |
| `no_gesture` | 0.727 | 0.061 | 0.112 | 660 |
| `ok_sign` | 0.706 | 0.996 | 0.826 | 460 |
| `open_palm` | 0.996 | 0.967 | 0.981 | 460 |
| `peace` | 0.877 | 1.000 | 0.935 | 500 |
| `point` | 0.673 | 1.000 | 0.805 | 460 |
| `rock_on` | 0.959 | 0.928 | 0.943 | 500 |
| `thumbs_up` | 1.000 | 0.996 | 0.998 | 720 |

Accuracy by handedness — Left: 0.868, Right: 0.833.

### By recording session

| Session | Lighting | n | Accuracy | Gestures only | `no_gesture` recall |
|---|---|---:|---:|---:|---:|
| `2026-10-12-svl1` | artificial | 2400 | 0.880 | 0.997 | 0.125 |
| `2026-10-13-s8lv` | daylight | 2280 | 0.821 | 0.964 | 0.000 |

![Confusion matrix](confusion_matrix.png)

Inference: 0.0009 ms/sample (Python, not in-browser). Fit time: 4.33 s.

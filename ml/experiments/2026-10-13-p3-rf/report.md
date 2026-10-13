# P3 — `rf`

**Question:** Does a model learned from HaGRID photos work on live webcam video in the app?

- Train: HaGRID train subjects
- Test: own_v2 (live webcam)
- Date: 2026-10-13
- Reproduce: `uv run python -m ml.eval --protocol p3 --model rf`

## Datasets

| Dataset | SHA-256 | Records used (raw) | Duplicates removed |
|---|---|---:|---:|
| `hagrid_landmarks_v1` (`ml/data/hagrid_landmarks_v1/records.jsonl`) | `d5db335e537d…` | 12485 (12495) | 10 |
| `own_v2` (`ml/data/own_v2.json`) | `86b72fad77f7…` | 4680 (4680) | 0 |

## Results

4680 test samples, 136 independent groups (bootstrap unit). Trained on 8362 samples from 6753 subjects.

| Metric | Value | 95 % CI |
|---|---:|---|
| accuracy | 0.854 | [0.798, 0.903] |
| macro_f1 | 0.824 | [0.777, 0.855] |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `call_me` | 0.956 | 0.950 | 0.953 | 460 |
| `fist` | 0.740 | 1.000 | 0.850 | 460 |
| `no_gesture` | 0.860 | 0.056 | 0.105 | 660 |
| `ok_sign` | 0.918 | 0.993 | 0.954 | 460 |
| `open_palm` | 0.758 | 1.000 | 0.862 | 460 |
| `peace` | 0.870 | 1.000 | 0.930 | 500 |
| `point` | 0.704 | 1.000 | 0.827 | 460 |
| `rock_on` | 0.959 | 0.936 | 0.947 | 500 |
| `thumbs_up` | 0.973 | 0.996 | 0.984 | 720 |

Accuracy by handedness — Left: 0.864, Right: 0.842.

### By recording session

| Session | Lighting | n | Accuracy | Gestures only | `no_gesture` recall |
|---|---|---:|---:|---:|---:|
| `2026-10-12-svl1` | artificial | 2400 | 0.879 | 0.997 | 0.116 |
| `2026-10-13-s8lv` | daylight | 2280 | 0.827 | 0.972 | 0.000 |

![Confusion matrix](confusion_matrix.png)

Inference: 0.0086 ms/sample (Python, not in-browser). Fit time: 6.35 s.

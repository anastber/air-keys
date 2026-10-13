# P3 — `logreg`

**Question:** Does a model learned from HaGRID photos work on live webcam video in the app?

- Train: HaGRID train subjects
- Test: own_v2 (live webcam)
- Date: 2026-10-13
- Reproduce: `uv run python -m ml.eval --protocol p3 --model logreg`

## Datasets

| Dataset | SHA-256 | Records used (raw) | Duplicates removed |
|---|---|---:|---:|
| `hagrid_landmarks_v1` (`ml/data/hagrid_landmarks_v1/records.jsonl`) | `d5db335e537d…` | 12485 (12495) | 10 |
| `own_v2` (`ml/data/own_v2.json`) | `86b72fad77f7…` | 4680 (4680) | 0 |

## Results

4680 test samples, 136 independent groups (bootstrap unit). Trained on 8362 samples from 6753 subjects.

| Metric | Value | 95 % CI |
|---|---:|---|
| accuracy | 0.850 | [0.795, 0.899] |
| macro_f1 | 0.821 | [0.774, 0.854] |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `call_me` | 0.818 | 0.957 | 0.882 | 460 |
| `fist` | 0.844 | 1.000 | 0.915 | 460 |
| `no_gesture` | 0.870 | 0.061 | 0.113 | 660 |
| `ok_sign` | 0.740 | 0.996 | 0.849 | 460 |
| `open_palm` | 0.915 | 0.987 | 0.950 | 460 |
| `peace` | 0.888 | 1.000 | 0.941 | 500 |
| `point` | 0.672 | 1.000 | 0.803 | 460 |
| `rock_on` | 0.959 | 0.930 | 0.944 | 500 |
| `thumbs_up` | 1.000 | 0.976 | 0.988 | 720 |

Accuracy by handedness — Left: 0.868, Right: 0.831.

### By recording session

| Session | Lighting | n | Accuracy | Gestures only | `no_gesture` recall |
|---|---|---:|---:|---:|---:|
| `2026-10-12-svl1` | artificial | 2400 | 0.875 | 0.990 | 0.125 |
| `2026-10-13-s8lv` | daylight | 2280 | 0.825 | 0.970 | 0.000 |

![Confusion matrix](confusion_matrix.png)

Inference: 0.0003 ms/sample (Python, not in-browser). Fit time: 0.14 s.

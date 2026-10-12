# P3 — `rf`

**Question:** Does a model learned from HaGRID photos work on live webcam video in the app?

- Train: HaGRID train subjects
- Test: own_v2 (live webcam)
- Date: 2026-10-12
- Reproduce: `uv run python -m ml.eval --protocol p3 --model rf`

## Datasets

| Dataset | SHA-256 | Records used (raw) | Duplicates removed |
|---|---|---:|---:|
| `hagrid_landmarks_v1` (`ml/data/hagrid_landmarks_v1/records.jsonl`) | `d5db335e537d…` | 12485 (12495) | 10 |
| `own_v2` (`ml/data/own_v2.json`) | `fe1abff90fc9…` | 2400 (2400) | 0 |

## Results

2400 test samples, 81 independent groups (bootstrap unit). Trained on 8362 samples from 6753 subjects.

| Metric | Value | 95 % CI |
|---|---:|---|
| accuracy | 0.879 | [0.808, 0.937] |
| macro_f1 | 0.856 | [0.794, 0.896] |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `call_me` | 0.928 | 0.988 | 0.957 | 260 |
| `fist` | 0.760 | 1.000 | 0.864 | 260 |
| `no_gesture` | 0.925 | 0.116 | 0.206 | 320 |
| `ok_sign` | 0.928 | 0.988 | 0.957 | 260 |
| `open_palm` | 0.758 | 1.000 | 0.862 | 260 |
| `peace` | 1.000 | 1.000 | 1.000 | 260 |
| `point` | 0.760 | 1.000 | 0.864 | 260 |
| `rock_on` | 1.000 | 0.996 | 0.998 | 260 |
| `thumbs_up` | 1.000 | 1.000 | 1.000 | 260 |

Accuracy by handedness — Left: 0.887, Right: 0.872.

![Confusion matrix](confusion_matrix.png)

Inference: 0.0077 ms/sample (Python, not in-browser). Fit time: 4.79 s.

# P1 — `rf`

**Question:** Does the model generalize to people it never saw?

- Train: HaGRID train subjects
- Test: HaGRID test subjects
- Date: 2026-10-12
- Reproduce: `uv run python -m ml.eval --protocol p1 --model rf`

## Datasets

| Dataset | SHA-256 | Records used (raw) | Duplicates removed |
|---|---|---:|---:|
| `hagrid_landmarks_v1` (`ml/data/hagrid_landmarks_v1/records.jsonl`) | `d5db335e537d…` | 12485 (12495) | 10 |

## Results

2497 test samples, 1309 independent groups (bootstrap unit). Trained on 8362 samples from 6753 subjects.

| Metric | Value | 95 % CI |
|---|---:|---|
| accuracy | 0.979 | [0.973, 0.984] |
| macro_f1 | 0.979 | [0.973, 0.984] |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `call_me` | 0.992 | 0.956 | 0.974 | 274 |
| `fist` | 0.986 | 1.000 | 0.993 | 272 |
| `no_gesture` | 0.947 | 0.934 | 0.941 | 288 |
| `ok_sign` | 0.996 | 0.993 | 0.995 | 282 |
| `open_palm` | 0.993 | 0.989 | 0.991 | 284 |
| `peace` | 0.986 | 0.993 | 0.989 | 279 |
| `point` | 0.978 | 0.993 | 0.985 | 272 |
| `rock_on` | 0.989 | 0.982 | 0.985 | 275 |
| `thumbs_up` | 0.943 | 0.970 | 0.956 | 271 |

Accuracy by handedness — Left: 0.978, Right: 0.980.

### By demographic group

HaGRID meta, automatically estimated (FairFace / MiVOLO), so approximate. Groups under 100 samples are marked *.

| Attribute | Group | Accuracy | n |
|---|---|---:|---:|
| gender | female | 0.987 | 1348 |
| gender | male | 0.969 | 1149 |
| race | Asian* | 0.990 | 98 |
| race | Black | 0.971 | 205 |
| race | Indian* | 0.986 | 72 |
| race | White | 0.981 | 1984 |
| race | unknown | 0.942 | 138 |
| age | 0-9* | 1.000 | 4 |
| age | 10-19 | 0.978 | 359 |
| age | 20-29 | 0.981 | 837 |
| age | 30-39 | 0.976 | 777 |
| age | 40-49 | 0.984 | 376 |
| age | 50-59 | 0.965 | 114 |
| age | 60+* | 1.000 | 30 |

![Confusion matrix](confusion_matrix.png)

Inference: 0.0147 ms/sample (Python, not in-browser). Fit time: 5.69 s.

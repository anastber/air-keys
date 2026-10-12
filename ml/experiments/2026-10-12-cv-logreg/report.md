# CV — `logreg`

**Question:** Model selection: grouped cross-validation on HaGRID training subjects.

- Train: HaGRID train subjects
- Test: none (cross-validation)
- Date: 2026-10-12
- Reproduce: `uv run python -m ml.eval --protocol cv --model logreg`

## Datasets

| Dataset | SHA-256 | Records used (raw) | Duplicates removed |
|---|---|---:|---:|
| `hagrid_landmarks_v1` (`ml/data/hagrid_landmarks_v1/records.jsonl`) | `d5db335e537d…` | 12485 (12495) | 10 |

## 5-fold grouped cross-validation

8362 samples from 6753 subjects; folds never share a subject.

| Metric | Mean ± std |
|---|---|
| accuracy | 0.987 ± 0.002 |
| macro_f1 | 0.987 ± 0.002 |

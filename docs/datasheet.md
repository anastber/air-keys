# Datasheet — AirKeys gesture datasets

Structure loosely follows *Datasheets for Datasets* (Gebru et al., 2021). This document covers the two datasets used by the ML pipeline (see [ROADMAP.md](../ROADMAP.md), Phase 0):

| Dataset | Content | Role |
|---|---|---|
| [`hagrid_landmarks_v1`](#hagrid_landmarks_v1) | Hand landmarks extracted from a sample of HaGRID, many subjects | Training + multi-subject test (protocols P1, P2) |
| [`own_v2`](#own_v2) | Hand landmarks recorded by the author through the app | Deployment-condition test (protocol P3) |

Neither dataset is committed to git: both stay in `ml/data/` (gitignored). They are identified by the SHA-256 of their file, which every validation and experiment report records. Datasets are immutable: any change produces a new version.

Statistics below come from `make validate DATA=<file>` (`ml/validation.py`), computed on the records as experiments consume them (`ml.data.load_dataset`: loaded, then deduplicated).

---

## `hagrid_landmarks_v1`

### Motivation
Measuring whether a gesture classifier generalizes to people it was not trained on requires many different hands. This dataset provides them without recruiting participants, by reusing a public dataset.

### Source and license
- **Source:** [HaGRID — HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid) (Kapitanov et al., [arXiv:2206.08219](https://arxiv.org/abs/2206.08219)), images and annotations from the HaGRIDv2 release.
- **License:** a variant of Creative Commons Attribution-ShareAlike 4.0 ([license text](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)): use, including commercial use, is allowed with attribution; adapted material must be shared under the same or a BY-SA-compatible license.
- **Changes made:** images were not redistributed or modified; hand landmarks were extracted from them (see below), and HaGRID labels were renamed to project labels.

### Composition

| | |
|---|---|
| File | `ml/data/hagrid_landmarks_v1/records.jsonl` |
| SHA-256 | `d5db335e537d9a85b401307a1c6cfbbefbd01ae42be33a88d41fdd05fdebdb33` |
| Records (raw / after deduplication) | 12,495 / 12,485 |
| Distinct subjects | 9,034 |
| Classes | 9 |
| Handedness (MediaPipe) | 6,817 right / 5,668 left |

One record = one hand in one image: 21 MediaPipe landmarks (`x`, `y`, `z`), handedness, project label, HaGRID label, HaGRID `user_id` and split, the annotated hand box, and HaGRID's own 2D landmarks for that hand. Record format: see `ml/data.py`.

Samples per class and split (after deduplication):

| Label (HaGRID class) | train | val | test |
|---|---:|---:|---:|
| `ok_sign` (`ok`) | 968 | 191 | 282 |
| `rock_on` (`rock`) | 918 | 183 | 275 |
| `call_me` (`call`) | 923 | 188 | 274 |
| `fist` (`fist`) | 929 | 182 | 272 |
| `open_palm` (`palm`) | 962 | 192 | 284 |
| `peace` (`peace`) | 941 | 182 | 279 |
| `point` (`one`) | 909 | 181 | 272 |
| `thumbs_up` (`like`) | 919 | 181 | 271 |
| `no_gesture` (`no_gesture`) | 893 | 146 | 288 |
| **Total** | **8,362** | **1,626** | **2,497** |

### Collection process
Produced by `ml/datasets/hagrid.py` (manifest: `ml/data/hagrid_landmarks_v1/manifest.json`):

1. **Sampling.** Within HaGRID's official train/val/test split — which is subject-disjoint — up to 1,000 / 200 / 300 images per class, at most 2 per subject, visiting subjects round-robin in a seeded order (seed 42) to maximize subject diversity. Images with zero or several hands of the target class are skipped (ambiguous).
2. **Reading.** Sampled images are read directly out of HaGRID's remote ~40 GB archives (HTTP range requests); no image is stored.
3. **Landmark extraction.** MediaPipe `GestureRecognizer` (Python, v1.1.0) with the same `gesture_recognizer.task` model as the frontend, in IMAGE mode, CPU, `num_hands=2`.
4. **Hand matching.** The detected hand whose landmark box best overlaps the annotated box is kept (IoU ≥ 0.3).
5. **Failures.** 882 of 13,377 sampled images yielded no usable hand and are logged in `failures.jsonl` with a reason: 708 `no_hand_detected`, 174 `no_matching_hand`. Success rate per class ranges from 90.8 % (`point`) to 96.5 % (`no_gesture`).

### Preprocessing and cleaning
- **Deduplication** (`ml.data.duplicates_to_drop`, applied by `load_dataset`): 10 records removed. HaGRID contains a few identical or near-identical scenes filed under *different* `user_id`s, some of them in different splits — a leak across the official subject-disjoint split. Exact landmark copies are dropped after the first; near-duplicates (normalized feature distance < 0.05, where the median nearest-neighbor distance is ~0.52) in different splits lose their val/test copy, so evaluation sets never contain a near-copy of a training sample.
- **Not filtered (yet):** 77 records (0.6 %) whose landmarks disagree with HaGRID's own annotation by more than 15 % of hand size — mostly MediaPipe errors on hard views (e.g. a back-of-hand OK sign). They are kept and flagged by validation; whether to filter them is an experiment decision, to be reported where made.

### Known limitations and biases
- **Still photos, not video.** No motion and no frame-to-frame tracking; MediaPipe runs in IMAGE mode, while the app runs it in VIDEO mode on a webcam stream. Protocol P3 measures the effect of this shift.
- **Demographics** (HaGRID meta, *automatically estimated* with FairFace / MiVOLO, so approximate; counted once per subject):
  - Gender: 51.5 % female, 48.4 % male.
  - Race: 75.8 % White, 7.9 % Black, 5.6 % Asian, 5.3 % Indian, 5.4 % unknown — strongly skewed; per-group performance should be checked before claiming the model works equally well for everyone.
  - Age: 20.5 % 10–19, 37.8 % 20–29, 26.2 % 30–39, 11.7 % 40–49, 3.7 % 50+ — older hands are under-represented.
- **Class imbalance.** `no_gesture` has fewer samples (HaGRID only has 1,464 such images, further reduced by the per-subject cap), most visibly in val (146 vs ~185).
- **Label definitions differ slightly from the app's.** HaGRID's `one` is an upward index finger (mapped to `point`), and its `no_gesture` contains natural resting hands only — not the hard negatives (e.g. pinch) that `own_v2` adds.
- **Subject IDs are imperfect,** as the duplicates above show; the deduplication removes the identifiable cases only.

### Uses
- Intended: training and evaluating hand-pose classifiers on MediaPipe landmarks, with subject-disjoint evaluation.
- Not intended: anything identifying or profiling people. Records keep HaGRID's pseudonymous `user_id` only to enforce subject-disjoint splits.

---

## `own_v2`

> Collection in progress: this section describes the protocol. Statistics are added once the planned sessions are recorded.

### Motivation
The only data captured in the app's real deployment condition — live webcam, in-browser MediaPipe (VIDEO mode, GPU delegate). Used only to measure how a model trained on HaGRID's photos transfers to actual use (protocol P3) — never as training data: HaGRID already covers far more people and conditions than the author's hands could add.

### Composition (planned)
- One contributor: the author. Consequently it cannot measure generalization to other people — that is `hagrid_landmarks_v1`'s role.
- The same 9 labels as `hagrid_landmarks_v1`; `no_gesture` additionally includes relaxed, curled and mid-transition hands, and pinches (which the app handles with geometry, not this model).
- 2 recording sessions in the conditions the app is actually played in (own room, own webcam, usual distance); both hands, varied positions. Representative rather than artificially varied conditions.

### Collection process
Recorded with the `/collect` dev page (`frontend/src/components/DatasetCollector.tsx`), which runs the production MediaPipe pipeline. Each Record click captures a burst of 20 consecutive frames. Every record carries `session_id`, `camera` (device name), `resolution` (as granted by the browser), and `lighting` (`daylight`, `artificial`, `dim` or `backlit`). Schema: see `ml/data.py`.

### Splitting
None needed: the whole dataset is a test set and is never trained on, so it cannot leak into training. Sessions are still tagged, so results can be broken down by session and lighting, and so a train/test split by session remains possible if fine-tuning is tried later.

### Known limitations
- Single person, single movement style, few cameras.
- Held poses only: transitions between gestures are recorded separately in Phase 1 (continuous sessions).
- Records from the earlier `gestures_v1` export (pre-v2, no session) cannot be used in session-disjoint evaluation; validation flags them.

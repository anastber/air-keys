# AirKeys — ML Roadmap

This document plans the next stage of AirKeys: moving from a working demo to a project with a rigorous machine-learning and computer-vision methodology. Every phase ships a **user-facing feature** and is driven by a **testable question**, a **baseline**, **pre-registered metrics**, and an explicit **go / no-go criterion**.

It started as a plan; results are added as tasks complete. Numbers in "success criteria" are provisional targets, to be revisited once Phase 0 has produced honest baselines.

Progress is tracked with the checkboxes below; a checked item references the pull request that delivered it.

---

## Where the project stands

| Component | Current approach | Known limitation |
|---|---|---|
| Base gestures (5) | MediaPipe `GestureRecognizer`, fixed 0.5 score threshold | Threshold not tuned; no temporal reasoning |
| Pinch | Thumb–index distance < 0.06 | Hand-picked threshold, scale-sensitive near the camera edges |
| Self-trained poses (3 + background) | MLP (32, 16) on normalized landmarks, burst-grouped split | **Single subject, single session** — 99.8 % says nothing about other hands |
| Personal gestures | Client-side kNN on raw normalized landmarks, ~20 samples | Needs many samples; distance threshold not tuned |
| Decision layer | Per-frame argmax + fixed priority order | No smoothing → jitter and spurious notes |
| Backend | Stub | No role yet |

The most important gap is the first line of the "known limitation" column for the MLP: **the current evaluation measures memorization of one person's hand, not generalization.** Phase 0 exists to fix that before anything else is built on top of it.

---

## Methodology (applies to every phase)

These rules are the "ML process" of the project. A phase is not done until its deliverables meet them.

1. **Question first.** Each experiment starts from a written hypothesis ("a GRU beats DTW on dynamic gestures under subject shift"), not from a model choice.
2. **Baseline before model.** Always compare against the simplest reasonable method (heuristic, kNN, logistic regression, DTW). A complex model that does not beat its baseline does not ship.
3. **Split by the unit of generalization.** Never split samples that share a recording burst, a session or a subject. On public data, splits are **subject-disjoint** (grouped by the dataset's subject ID); on the project's own recordings, they are **session-disjoint**. The burst-grouped split is kept as a secondary sanity check.
4. **Test set is touched once.** Model selection and hyperparameter search happen with grouped cross-validation on the training subjects. The held-out test subjects — and the own-webcam test sessions — are evaluated only at the end of a phase.
5. **Report uncertainty.** Metrics are reported as mean ± std across grouped CV folds, with bootstrap 95 % CIs (resampling subjects, not samples) on the test set — never as a single number.
6. **Metrics beyond accuracy.** Macro-F1, per-class recall, confusion matrix, calibration (ECE), and — because this is an instrument — **latency** and **false triggers per minute**.
7. **Reproducibility.** Fixed seeds, versioned datasets with a content hash, the full configuration recorded in each report, and one command to rerun it. Each experiment writes a short report to `ml/experiments/<id>/`.
8. **Train/serve parity.** Preprocessing is implemented in both Python and TypeScript; a parity test on shared fixtures guarantees they produce identical features.
9. **Error analysis is a deliverable.** Each phase ends with a written look at failure cases, not only aggregate scores.
10. **Model cards.** Every deployed model ships with a card: intended use, training data, evaluation protocol, known failure modes.

---

## Phase 0 — Evaluation foundations

> **Question:** How well do the current models generalize to people who were not in the training data?

This phase builds no new model. It builds the infrastructure that every later phase depends on.

### Data strategy: public multi-subject data, no recruitment

Measuring generalization to other people requires other people's hands — but not people recruited by this project. Multi-subject data comes from **[HaGRID](https://github.com/ai-forever/hagrid)**, a public hand-gesture image dataset with tens of thousands of distinct subjects, varied cameras and lighting, and a `user_id` per image. It covers the three self-trained classes (`ok`, `rock`, `call`), most base gestures (`fist`, `palm`, `peace`, `one`), and a `no_gesture` class.

The project's own recordings are kept, with a different role: they are the only data captured in the **real deployment condition** (live webcam video through the in-browser pipeline).

| Dataset | Source | Subjects | Role |
|---|---|---|---|
| `hagrid_landmarks` | HaGRID images → MediaPipe | Many, subject-disjoint splits | Training + **multi-subject test** |
| `own_v2` | `/collect`, author only, 2 sessions | 1 | **Deployment-condition test** only — never trained on |

This makes three evaluation protocols possible, each answering a different question:

| Protocol | Train → Test | Question it answers |
|---|---|---|
| **P1 — in-domain** | HaGRID train subjects → HaGRID test subjects | Does the model generalize to unseen people? |
| **P2 — cross-dataset** | Own data → HaGRID test subjects | Does a model trained on one hand generalize to thousands? (the honest test of the current MLP) |
| **P3 — deployment shift** | HaGRID → own recordings | Does a model learned from photos work on live webcam video? |

Known limitation, stated up front: HaGRID consists of still photos, not video, so it cannot evaluate temporal behavior (Phase 1) and does not contain motion gestures (Phase 2).

### 0.1 Landmark extraction from HaGRID
- [x] Restrict to the relevant classes and extract a **subject-stratified sample** within HaGRID's official subject-disjoint train/val/test split. The archives are ~40 GB per class, so sampled images are read straight out of the remote zips (HTTP range requests) and never stored — only landmarks are kept. *(#4)*
- [x] **Re-extract landmarks with MediaPipe Python, using the same `.task` model as the frontend**, rather than using HaGRID's provided landmarks (2D only, no `z`, different extraction settings). This keeps train/serve preprocessing identical. *(#4)*
- [x] Log and analyze extraction failures (no hand detected, wrong hand selected in multi-hand images): failure rate per class is itself a finding, and silently dropping failures would bias the dataset. *(#4: logged with a reason per image; analysis continues in 0.3)*
- [x] Map HaGRID labels to project labels (`ok` → `ok_sign`, `rock` → `rock_on`, `call` → `call_me`) in one explicit, versioned mapping file. *(#4: `ml/datasets/hagrid_labels.json`, 9 classes)*
- [x] Verify and document the dataset's license before publishing any results or weights trained on it. *(#4: variant of CC BY-SA 4.0, attribution + share-alike)*
- [x] Add `mediapipe` to the offline dependencies (training only; it is not a runtime dependency). *(#4)*

> **Status — done.** `hagrid_landmarks_v1`: 12,495 hand samples from 9,520 distinct subjects (9 classes, up to 1000/200/300 images per train/val/test split, ≤ 2 per subject), zero subject overlap between splits. Extraction success 90.8–96.5 % per class; median gap to HaGRID's own landmarks 1.6 % of hand size.

### 0.2 Own-data protocol (`own_v2`)
- [x] Extend the `/collect` record schema with `session_id`, camera / resolution, and a lighting tag.
- [ ] Record **2 sessions** in the conditions the app is actually played in (own room, own webcam, usual distance), both hands.
- [ ] Collect **hard negatives** explicitly: half-formed poses, hands at rest, hands holding objects, pinches.

> **Why only 2 sessions.** `own_v2` is a test set only: HaGRID already brings far more variety in people, rooms, light and cameras than a few extra sessions could, so the author's hands add little as *training* data. What HaGRID cannot provide is the deployment condition itself — live webcam video, MediaPipe in VIDEO mode, the author's real setup — and that is what `own_v2` measures. Since it is never trained on, train/test separation is guaranteed by construction; representative conditions matter more than artificially varied ones. Fine-tuning on own data may come back later as an optional experiment, with more sessions.
>
> Transitions between gestures — the main source of wrong notes when playing, absent from both HaGRID and held poses — are recorded in Phase 1 as continuous sessions.

### 0.3 Dataset versioning and validation
- [x] Datasets are immutable files, identified by the SHA-256 of their content (computed by the validation report). Raw data stays out of git (as today).
- [x] Record the dataset SHA-256 in each experiment report (with the 0.4 harness).
- [x] A validation script checks schema, subject leakage across splits, class balance per split, duplicates and landmark quality (`make validate DATA=<file>`, `ml/validation.py`).
- [x] Deduplicate before any experiment (`ml.data.load_dataset`): HaGRID files a few identical scenes under different `user_id`s, some across splits — exact copies and cross-split near-duplicates are removed from val/test.
- [x] Credit HaGRID (authors, link, license) in the README and the datasheet
- [x] A **datasheet** documents both datasets: sources, label mapping, extraction pipeline, known biases (e.g. HaGRID's demographic and camera distribution) — `docs/datasheet.md`; the `own_v2` statistics are added once its sessions are recorded.
- [ ] Exploratory analysis notebook: per-class distributions, PCA/UMAP of features colored by dataset — a visible gap between HaGRID and own-webcam clusters quantifies the domain shift that P3 measures. *(waits for `own_v2`)*

> **Status — `hagrid_landmarks_v1` validated.** No errors. 10 duplicates removed (all from val/test), leaving 12,485 records from 9,034 subjects. Warnings kept for experiments to handle: 77 records (0.6 %) with likely landmark errors, fewer `no_gesture` samples. Demographics skew strongly White (75.8 %) and young (< 40: 84.5 %), to be checked per group in evaluation.

### 0.4 Evaluation harness
- [x] Refactor `ml/train.py` into `ml/data.py` (loading, grouping, splits) and `ml/evaluate.py` (metrics, reports) *(#3)*
- [x] Candidate models in one registry (`ml/models.py`: logistic regression, kNN, random forest, MLP), plus the deployed MLP reloaded from its exported weights and run exactly as the frontend does (same forward pass, threshold and background class — tested against sklearn)
- [x] Protocols P1–P3 (`ml/protocols.py`) and grouped 5-fold CV on HaGRID training subjects for model selection. HaGRID's val split is reserved for Phase 1 calibration. `own_v2` is test-only, so P3 needs no own-data split; its CIs resample recording bursts.
- [x] Single entry point: `make eval PROTOCOL=p1 MODEL=mlp` → validates the datasets, then writes `metrics.json`, `report.md` and a confusion matrix to `ml/experiments/<date>-<protocol>-<model>/`, with bootstrap 95 % CIs over subjects, per-class metrics and the command that reproduces the run.
- [x] Add Python ↔ TypeScript **feature parity test** on fixture landmarks (`ml/features.py` vs `lib/customGestures.ts::normalizeLandmarks`), extended to the deployed MLP's forward pass (`ml/models.py` vs `lib/trainedGestures.ts`) — shared fixture `tests/fixtures/parity.json`, run in CI on both sides.

### 0.5 Re-baseline existing models
- [x] Evaluate the current deployed MLP under **P2** — the honest version of today's 99.8 %.
- [x] Select a model by grouped CV on HaGRID training subjects only (rule fixed beforehand: best CV macro-F1, ties within one standard deviation go to the simplest model).
- [x] Train and evaluate RandomForest, logistic regression, kNN and the MLP under P1 and P3.
- [x] Break P1/P2 results down by demographic group (HaGRID's estimated gender, ethnicity, age).
- [x] Measure inference time per frame for each model — in V8, Chrome's JS engine, with the app's own code paths (`npm run bench`).
- [x] Update the README with the new numbers (expect P2 to drop — that is the point).
- [ ] Re-run P3 once `own_v2`'s second session is recorded.

> **Status — done** (all runs: [`ml/experiments/README.md`](ml/experiments/README.md)).
> - **CV selects logistic regression** (macro-F1 0.987 ± 0.002); MLP (0.986 ± 0.004), kNN and random forest (0.980) are tied or close, so the simplest model wins.
> - **P1:** logistic regression 0.980 [0.974, 0.985] on unseen HaGRID subjects (MLP 0.985 [0.980, 0.989] — overlapping, and switching after seeing the test set would defeat its purpose). Uniform across demographic groups (0.97–0.99).
> - **P2:** the deployed MLP drops from 99.8 % (its author's hand) to **0.719** [0.702, 0.736]; `call_me` false-triggers (precision 0.36). On the same 4-class task, logistic regression scores 0.990.
> - **P3** (1 session): **0.990 on the 8 gestures** — the photo → webcam shift is small — but `no_gesture` recall is **0.125**: resting, half-formed and pinching hands are taken for `point`, `ok_sign` or `fist`. HaGRID's `no_gesture` only covers natural resting hands. This makes open-set rejection (Phase 1) the priority before replacing the deployed model.
> - **Inference** (V8, per frame): normalization 0.6 µs, logistic regression 2 µs, MLP 6 µs, kNN over the training set 0.9 ms — all far below a 33 ms frame; kNN would also have to ship 8,362 training samples.

**Deliverables:** HaGRID extraction pipeline, `own_v2`, datasheet, evaluation harness, parity test, re-baseline report.
**Done when:** any model can be evaluated under P1–P3 with one command, and the README reports subject-independent results.

---

## Phase 1 — Temporal stability and note onset

> **Question:** Can we cut spurious notes substantially without adding perceptible latency?

For an instrument, a wrong note or a late note is worse than a missed gesture. Today every frame is classified independently.

### Data
- [ ] HaGRID's still photos cannot evaluate temporal behavior, so this phase uses the author's own recordings: **continuous sessions** (30–60 s) playing scripted sequences (e.g. the Songs tab melodies), with timestamped ground-truth labels for each segment (annotated after the fact with a simple labeling tool on the landmark stream). Sessions are split session-disjoint.
- [ ] Limitation, documented in the report: temporal parameters are tuned on one person's movement style.

### Metrics
- [ ] **False triggers per minute** (note-on events not matching ground truth).
- [ ] **Onset latency**: time between ground-truth gesture start and note-on (median and p95, in ms).
- [ ] **Missed onsets** rate.
- [ ] Plotted as a trade-off curve: every smoothing method is a point on a latency / false-trigger plane.

### Experiments
- [ ] **Baseline:** current per-frame argmax.
- [ ] **Landmark filtering:** One-Euro filter vs. constant-velocity Kalman filter on landmark coordinates. Tune on training sessions.
- [ ] **Probability calibration:** temperature scaling on the MLP, fitted on HaGRID validation subjects; reliability diagrams and ECE before/after, on both HaGRID test subjects and own sessions (calibration often degrades under domain shift — worth measuring). Makes thresholds meaningful and comparable across models.
- [ ] **Decision smoothing:** majority vote over a window, hysteresis (separate on/off thresholds), and an HMM / forward filter over class posteriors with a learned transition matrix.
- [ ] **Open-set rejection** — *the priority after Phase 0.5: a HaGRID-trained model rejects only 12.5 % of the author's resting/half-formed hands.* Compare max-softmax threshold, energy score, and distance-to-class-centroid for rejecting "unknown" hand shapes. Evaluate with AUROC on held-out hard negatives — HaGRID's unused classes (`four`, `mute`, `three`, …) provide a large, multi-subject source of realistic "other gesture" samples.
- [ ] **Threshold tuning:** replace the hand-picked thresholds (MediaPipe 0.5, MLP 0.6, pinch 0.06) with values chosen on dev data.

### Feature
- [ ] Cleaner playing, plus a "responsiveness" control in the UI that moves along the measured trade-off curve.

**Done when:** a method dominates the baseline on the trade-off curve (target: ≥ 50 % fewer false triggers at ≤ +30 ms median latency), and is deployed with calibrated thresholds.

---

## Phase 2 — Dynamic gesture recognition

> **Question:** Can sequence models recognize motion gestures robustly across subjects, and which architecture gives the best accuracy / latency trade-off in the browser?

### Target gestures
Swipe left / right (octave or preset change), circle (loop / sustain), shake (vibrato), plus a background class of non-gesture motion.

### Data
HaGRID has no motion, so multi-subject data comes from public dynamic-gesture datasets instead — still without recruiting anyone:
- [ ] **Skeleton-based:** SHREC'17 / DHG-14/28 — hand-skeleton sequences from many subjects, including swipes, rotations and shakes. Closest to this project's landmark representation.
- [ ] **Video-based:** IPN Hand (and similar) — run through MediaPipe to obtain landmark sequences with the deployment pipeline.
- [ ] **Own recordings:** windowed landmark sequences of the target gestures, with start/end annotations — the deployment-condition test set, as in Phase 0.
- [ ] Licenses of these datasets are often research-only: each is checked before use, and this decides whether trained weights can ship in the public app or only appear in the experiment report.
- [ ] Skeleton formats differ from MediaPipe's 21-landmark layout: a documented joint-mapping step converts them, and its effect is measured (train on mapped public data, test on own recordings).
- [ ] **Augmentation:** time warping, speed scaling, small rotations / scalings, Gaussian jitter, random cropping of the window. Each augmentation evaluated by ablation, not assumed to help.

### Models (in order of complexity)
- [ ] **Baseline 0 — static pose + trajectory rules,** as in HaGRID's own [dynamic gestures](https://github.com/ai-forever/dynamic_gestures) work ([HaGRIDv2 paper](https://arxiv.org/abs/2412.01508)): a dynamic gesture is a start pose, a movement and an end pose, recognized by a state machine over the Phase 0 static classifier's predictions and the hand's trajectory. Needs no video training data at all; MediaPipe already provides the hand position and tracking. The learned models below — and the public video datasets they need — only matter if they beat it. Unlike the original, its thresholds are tuned and evaluated on data rather than set by hand.
- [ ] **DTW + kNN** — strong, interpretable baseline for few-sample motion recognition.
- [ ] **Hand-crafted temporal features** (velocities, displacement, trajectory curvature) + gradient boosting.
- [ ] **1D temporal CNN** on landmark windows.
- [ ] **GRU** (small, unidirectional — causal so it can run in real time).
- [ ] Optional: small **Transformer** encoder, only if 3–4 plateau.

### Segmentation
- [ ] Sliding-window classification with a background class vs. an explicit start/end detector. Evaluate with segment-level F1 (IoU-matched), not window accuracy.

### Deployment
- [ ] The hand-rolled forward pass stops scaling here. Introduce **ONNX export + ONNX Runtime Web**, with an end-to-end test that browser outputs match Python outputs on fixtures.
- [ ] Latency budget: model inference ≤ 5 ms per frame on a mid-range laptop.

**Done when:** the selected model beats DTW on subject-disjoint segment-level F1 with non-overlapping confidence intervals, fits the latency budget, and ships as musical effects in the app.

---

## Phase 3 — Few-shot personal gestures via metric learning

> **Question:** Does a learned embedding let users teach a new gesture from 3–5 samples with better accuracy than kNN on raw features?

### Approach
- [ ] Train an embedding network offline with **triplet loss** and **prototypical networks**, episodic training over HaGRID landmarks. HaGRIDv2's 30+ gesture classes and many subjects make it well suited to episodic few-shot training.
- [ ] At inference, the client-side kNN stays — but operates in the learned embedding space instead of on raw normalized landmarks. Personal data still never leaves the browser.

### Evaluation protocol
- [ ] **Class-disjoint split:** a fixed subset of HaGRID classes is never seen during embedding training, combined with subject-disjoint splits.
- [ ] **Deployment check:** the author teaches new gestures through the real UI; K-shot accuracy measured on held-out own sessions.
- [ ] **N-way K-shot accuracy** for K ∈ {1, 3, 5, 10, 20}, compared against raw-feature kNN at the same K.
- [ ] **Open-set:** rejection of gestures the user never taught (AUROC).
- [ ] Embedding visualization (UMAP) for qualitative inspection.

### Feature
- [ ] "Teach a gesture" needs 3–5 samples instead of ~20, and gives a live confidence indicator while recording.

**Done when:** the embedding approach beats raw kNN at K = 5 on unseen classes and unseen subjects.

---

## Phase 4 — Continuous expressive control

> **Question:** Can a learned low-dimensional representation of hand pose provide smooth, controllable, continuous musical parameters?

### Approach
- [ ] **PCA** on normalized landmarks as the baseline latent space.
- [ ] **Autoencoder / VAE** with a 2–3-dimensional latent space.
- [ ] Map latent coordinates to synthesis parameters (filter cutoff, timbre morph, vibrato depth).

### Evaluation
- [ ] Reconstruction error (held-out subjects).
- [ ] **Smoothness:** latent-space jitter while holding a static pose vs. displacement during intentional movement (signal-to-noise ratio).
- [ ] **Controllability:** a scripted target-reaching task (in the spirit of Fitts's law experiments) — hit target parameter values, measure time-to-target and error over many trials. Compare against simple geometric controls (hand openness, wrist angle) as a baseline. Run by the author only, so results are reported as a single-user evaluation, not a user study.

### Feature
- [ ] An "expressive mode": the hand acts as a continuous timbre pad on top of discrete note selection.

**Done when:** the learned control beats geometric controls on the user-study controllability metrics, or the phase concludes — with data — that geometry is enough (a valid outcome, documented as such).

---

## Phase 5 — Pixel-based vision

> **Question:** Does an image model on hand crops add robustness where landmarks fail (occlusion, poor lighting, overlapping hands), and at what latency cost?

### Data
- [ ] **HaGRID images directly** — the same images used for landmark extraction in Phase 0, cropped around the hand (HaGRID's bounding boxes). No image collection from other people. This also enables a clean comparison: landmark model and image model are trained and tested on the exact same subjects and samples.
- [ ] A **hard-conditions test set**, built two ways: (a) HaGRID images where MediaPipe landmark extraction failed or had low confidence (logged in Phase 0.1) — precisely the cases a pixel model should help with; (b) the author's own webcam crops in low light, backlight, motion blur and partial occlusion.

### Experiments
- [ ] Transfer learning on **MobileNetV3-Small / EfficientNet-Lite** (frozen backbone → partial fine-tuning).
- [ ] Head-to-head comparison against the landmark MLP on standard vs. hard-conditions test sets.
- [ ] **Late fusion** of landmark and image models; evaluate whether fusion improves hard-conditions accuracy without hurting the standard set.
- [ ] **Explainability:** Grad-CAM maps to verify the model looks at the hand, not the background (shortcut-learning check).
- [ ] **Efficiency:** int8 quantization, WebGPU vs. WASM execution; accuracy drop vs. speedup.

### Feature
- [ ] Image model used as a fallback when landmark confidence is low — only if it earns its place.

**Done when:** a documented comparison exists. Shipping is conditional: the image model ships only if it improves hard-conditions accuracy within the latency budget. A negative result, properly reported, is an acceptable outcome.

---

## Phase 6 — Data loop and model lifecycle

> **Question:** Can the models improve over time from real usage, without compromising user privacy?

This phase gives the backend its role. It is the only phase that involves other people's data beyond public datasets, and is therefore **optional**: the lifecycle pieces (pipeline, gating, registry, monitoring) are valuable on their own, fed by public data and own recordings.

- [ ] **Opt-in landmark donation (optional):** anonymous, landmarks only, with clear consent UI and deletion support — would require a GDPR review before going live.
- [ ] **Active learning:** prompt users to confirm predictions where the model is least confident (uncertainty / margin sampling); compare against random sampling for label efficiency.
- [ ] **Training pipeline:** reproducible retraining from versioned data, gated by the Phase 0 evaluation harness (a new model must beat the current one on the fixed P1 and P3 test sets).
- [ ] **Model registry:** versioned models with their model cards and evaluation reports; the frontend loads a pinned version.
- [ ] **Monitoring:** feature-distribution drift (e.g. population stability index on landmark features), confidence distribution over time, rejection rate.
- [ ] **Controlled rollout:** shadow-mode evaluation before a new model replaces the current one.

**Done when:** a full cycle — new data → retrain → gated evaluation → versioned release — has run end to end at least once.

---

## Dependencies

```
Phase 0 (evaluation foundations)
   ├──> Phase 1 (temporal stability) ──> Phase 2 (dynamic gestures) ──> Phase 5 (pixel vision)
   ├──> Phase 3 (few-shot metric learning)
   └──> Phase 4 (continuous control)
                                    all ──> Phase 6 (data loop & lifecycle)
```

- Phase 0 is a hard prerequisite: the HaGRID pipeline and protocols P1–P3 are required by every later phase.
- Phase 1's continuous annotated recordings and windowing are reused by Phase 2.
- Phase 2 introduces ONNX Runtime Web, which Phase 5 relies on.
- Phases 3 and 4 depend only on Phase 0 and can run in parallel with 1–2.

---

## Repository layout

```
ml/
  data.py            # loading, deduplication, grouping, splits
  validation.py      # dataset checks (make validate)
  datasets/          # public-dataset adapters: HaGRID extraction, label mappings
  features.py        # normalization (parity-tested against the TS implementation)
  models.py          # candidate models + the deployed MLP
  protocols.py       # evaluation protocols P1–P3, CV
  evaluate.py        # metrics, bootstrap CIs, confusion matrices
  eval.py            # evaluation entry point (make eval)
  experiments/<id>/  # report.md, metrics.json, plots — committed
  notebooks/         # EDA and error analysis (planned)
  model_cards/       # one card per deployed model (planned)
docs/
  datasheet.md       # hagrid_landmarks + own_v2
```

---

## Out of scope (for now)

- Training a hand-landmark detector from scratch — MediaPipe's is strong, and the project's value lies in what is built on top of it.
- Server-side inference — client-side inference is a deliberate privacy and latency choice (see the README case study) and stays the default.
- Recruiting participants for data collection — multi-subject data comes from public datasets; the author's own recordings cover the deployment condition.
- Collecting images of anyone other than the author.

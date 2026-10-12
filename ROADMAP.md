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
7. **Reproducibility.** Fixed seeds, versioned datasets with a content hash, a config file per experiment, and one command to rerun it. Each experiment writes a short report to `ml/experiments/<id>/`.
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
| `own_v2` | `/collect`, author only, multiple sessions | 1 | Fine-tuning + **deployment-condition test** |

This makes three evaluation protocols possible, each answering a different question:

| Protocol | Train → Test | Question it answers |
|---|---|---|
| **P1 — in-domain** | HaGRID train subjects → HaGRID test subjects | Does the model generalize to unseen people? |
| **P2 — cross-dataset** | Own data → HaGRID test subjects | Does a model trained on one hand generalize to thousands? (the honest test of the current MLP) |
| **P3 — deployment shift** | HaGRID (± own train sessions) → own held-out sessions | Does a model learned from photos work on live webcam video? |

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
- [ ] Record **≥ 5 sessions** on different days, lighting conditions, distances and both hands; hold out whole sessions for testing.
- [ ] Collect **hard negatives** explicitly: transitions between poses, half-formed poses, hands at rest, hands holding objects.

### 0.3 Dataset versioning and validation
- [ ] Datasets are immutable files named by version + SHA-256 hash, recorded in each experiment report. Raw data stays out of git (as today).
- [ ] A validation script checks schema, class balance per split, missing landmarks, and duplicated bursts.
- [ ] Credit HaGRID (authors, link, license) in the README and the datasheet
- [ ] A **datasheet** documents both datasets: sources, label mapping, extraction pipeline, known biases (e.g. HaGRID's demographic and camera distribution).
- [ ] Exploratory analysis notebook: per-class distributions, PCA/UMAP of features colored by dataset — a visible gap between HaGRID and own-webcam clusters quantifies the domain shift that P3 measures.

### 0.4 Evaluation harness
- [x] Refactor `ml/train.py` into `ml/data.py` (loading, grouping, splits) and `ml/evaluate.py` (metrics, reports) *(#3)*
- [ ] Move model-specific training into their own modules as models are added
- [ ] Implement subject-disjoint grouped CV for HaGRID, session-disjoint splits for own data, and protocols P1–P3.
- [ ] Single entry point: `make eval MODEL=mlp PROTOCOL=p1` → writes metrics JSON, confusion matrices, and a markdown report.
- [ ] Add Python ↔ TypeScript **feature parity test** on fixture landmarks (`ml/features.py` vs `lib/customGestures.ts::normalizeLandmarks`).

### 0.5 Re-baseline existing models
- [ ] Evaluate the current deployed MLP under **P2** — the honest version of today's 99.8 %.
- [ ] Train and evaluate RandomForest, logistic regression, kNN and the MLP under P1 and P3.
- [ ] Measure in-browser inference time per frame for each model.
- [ ] Update the README with the new numbers (expect P2 to drop — that is the point).

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
- [ ] **Open-set rejection:** compare max-softmax threshold, energy score, and distance-to-class-centroid for rejecting "unknown" hand shapes. Evaluate with AUROC on held-out hard negatives — HaGRID's unused classes (`four`, `mute`, `three`, …) provide a large, multi-subject source of realistic "other gesture" samples.
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

## Planned repository layout

```
ml/
  data.py            # loading, validation, grouping, splits (burst / session / subject)
  datasets/          # public-dataset adapters: HaGRID extraction, label mappings
  protocols.py       # evaluation protocols P1–P3
  features.py        # normalization (parity-tested against the TS implementation)
  evaluate.py        # metrics, CIs, reports, confusion matrices
  models/            # one module per model family
  configs/           # one config per experiment
  experiments/<id>/  # report.md, metrics.json, plots — committed
  notebooks/         # EDA and error analysis
  model_cards/       # one card per deployed model
docs/
  datasheet.md       # hagrid_landmarks + own_v2
```

---

## Out of scope (for now)

- Training a hand-landmark detector from scratch — MediaPipe's is strong, and the project's value lies in what is built on top of it.
- Server-side inference — client-side inference is a deliberate privacy and latency choice (see the README case study) and stays the default.
- Recruiting participants for data collection — multi-subject data comes from public datasets; the author's own recordings cover the deployment condition.
- Collecting images of anyone other than the author.

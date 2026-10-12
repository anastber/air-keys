# AirKeys

**Camera-Based Gesture Instrument**

AirKeys turns your webcam into a musical instrument. Eight hand gestures each play a fixed note of a one-octave scale, so you can play real melodies (the in-app Songs tab has a few). You can also teach it gestures of your own.

Everything — hand tracking, gesture recognition, and audio — runs in your browser, with no account and no setup. Three of the gestures are recognized by a small neural net I trained myself (see [Self-trained gesture classifier](#self-trained-gesture-classifier)).

## How it works

```
Webcam --> MediaPipe (21 landmarks + pretrained gesture)  ---\
       --> pinch (pure geometry)                            |
       --> self-trained MLP (offline-trained, 3 poses)       >--> Rule Engine --> Tone.js audio
       --> your taught gestures (client-side kNN)          -/
                    [all in the browser]                    [gesture -> musical action]
```

- **Hand tracking + base gestures**: MediaPipe's pretrained `GestureRecognizer` extracts 21 hand landmarks and classifies 5 canned poses (fist, open palm, point, peace, thumbs up) in one model call, running fully client-side via WASM. It was trained by Google on a large, diverse hand dataset, so it works for any visitor immediately — no training step, no per-user setup.
- **Pinch**: not a model at all — a plain thumb-to-index-fingertip distance check. Geometry generalizes to any hand by construction, so this one didn't need ML.
- **Self-trained classifier**: OK sign, rock on, and call me are classified by a small MLP trained offline on landmark data I collected and labeled myself — a real train/test split, a confusion matrix, the works. Ships with the app; no per-visitor setup. Details and results below.
- **Personalization**: teach it a gesture of your own and it's classified by k-nearest-neighbors against samples you record, right there in the training panel. No training step beyond recording — a sample is added, that's it. Everything stays in your browser's `localStorage`; nothing is sent to a server.
- **Rule engine**: whichever gesture wins (taught gestures first, then the self-trained model, then pinch, then the canned poses) is mapped to a musical action — editable live in the UI, not fixed in code. See [Playing a melody](#playing-a-melody) for the default mapping.
- **Audio**: Tone.js synthesizes the result.

## Why it's built this way — a case study in getting this wrong first

The first version of gesture recognition here trained a random forest classifier **on the backend**, live, per visit: a visitor would record samples through the UI, hit train, and the server would fit a new model on the spot. It worked well as a single-person demo — real train/held-out-test evaluation, a confusion matrix, sub-second retraining.

It breaks the moment more than one person can use the app at once. The backend held **one shared model and one shared dataset file** for every visitor. A second visitor training their own gesture would silently retrain — and replace — the model the first visitor was mid-session with. There was no concept of "user" anywhere in that design. Fine for a personal demo running on one laptop; not something you can point a public URL at.

The fix wasn't more infrastructure (per-session backend isolation, cleanup jobs, etc.) — it was recognizing that most of what needs to be personalized doesn't need a server at all:

- **Base gestures** moved to a pretrained model (MediaPipe's `GestureRecognizer`) instead of something trained per-user, so there's no cold start and nothing to keep consistent across visitors.
- **Personalization** moved into the browser (`localStorage` + a from-scratch kNN classifier), so each visitor's taught gestures are private to them by construction — there's no shared state to corrupt because there's no shared state.
- As a side effect, the backend round-trip (WebSocket video streaming to a Python MediaPipe process) disappeared too, along with the latency it added — landmark extraction now happens directly against the `<video>` element.

The backend is now a stub — it doesn't do any gesture work at all. What it's used for next hasn't been decided yet.

## Self-trained gesture classifier

MediaPipe's pretrained recognizer and the client-side kNN cover 5+N poses without me training anything myself. To have an actual supervised-learning pipeline in this project — data collection, feature engineering, a real held-out evaluation — I added 3 more poses (`ok_sign`, `rock_on`, `call_me`) classified by a small MLP I trained offline, plus a `no_gesture` background class so it doesn't fire on a resting hand.

**Data**: recorded with the `/collect` dev page (`DatasetCollector.tsx`), which runs the exact same in-browser MediaPipe pipeline used at inference time and exports labeled landmark samples as JSON (`gestures_v1.json`). 2,120 samples total across the 4 classes, both hands, varying distance/rotation/position across ~15-19 separate recording bursts per class. (`/collect` has since been extended to 9 classes and per-session tagging for the next dataset — see [Datasets](#datasets).)

**Held-out results** (whole recording bursts are held out, so near-identical consecutive frames never end up on both sides of the split):

| Model | Held-out accuracy | Notes |
|---|---|---|
| RandomForest (baseline) | 96.6% | Confusion mainly between `ok_sign` and `no_gesture` |
| MLP, hidden layers (32, 16) — **deployed** | 99.8% | See `ml/experiments/confusion_matrix.png` |

**Caveat: every sample comes from one person (me), in one recording session.** These numbers measure how well the model recognizes *my* hand on 4 distinct static poses, not anyone's. The [roadmap](ROADMAP.md) re-evaluates it on thousands of other people's hands (HaGRID, protocol P2); this section will be updated with those results, which are expected to be lower.

**Feature engineering**: the same wrist-centered, hand-span-scaled, handedness-mirrored normalization already used by the client-side kNN (`lib/customGestures.ts::normalizeLandmarks`), ported to Python (`ml/features.py`) so train-time and inference-time preprocessing stay identical.

**Deployment, without an inference runtime**: the MLP is small enough that its weight matrices (`coefs_`/`intercepts_`) export straight to JSON, and `lib/trainedGestures.ts` hand-rolls the forward pass (two dense layers + ReLU + softmax) in ~30 lines of TypeScript — no ONNX Runtime Web or TensorFlow.js dependency, consistent with this app's existing precedent of hand-rolling classification client-side (the kNN does the same).

**Reproducing it**: `uv run python -m ml.train` (needs `ml/data/gestures_v1.json`, collected via `/collect`). The exported weights JSON and confusion matrix are committed to git as evidence + the deployable artifact; the raw dataset and full sklearn pickle aren't (privacy + size) — see `.gitignore`. `frontend/scripts/setup-mediapipe.mjs` copies the committed weights into the gitignored `public/` dir on every install.

## Datasets

The next stage of the ML work ([ROADMAP.md](ROADMAP.md)) evaluates models on people they were never trained on, using two landmark datasets, documented in the [datasheet](docs/datasheet.md):

- **`hagrid_landmarks_v1`** — 12,485 hands from 9,034 distinct people, extracted from [HaGRID](https://github.com/hukenovs/hagrid) with the app's own MediaPipe model (`uv run python -m ml.datasets.hagrid`). Images are read straight out of HaGRID's remote archives and never stored; only landmarks are kept.
- **`own_v2`** — the author's own recordings through `/collect`, tagged by session, camera and lighting: the only data captured in real webcam conditions.

Every dataset is checked before use (`make validate DATA=<file>`): schema, subject leakage across splits, class balance, duplicates and landmark quality, with the file's SHA-256 identifying the exact version. Neither dataset is committed to git.

## Tech Stack

### Frontend

- **Next.js 14 / TypeScript** - web interface
- **@mediapipe/tasks-vision** - pretrained hand landmark + gesture recognition, running in-browser via WASM
- **Tone.js** - web audio synthesis
- **TailwindCSS** - styling

### Backend

- **FastAPI** - currently a stub, not wired to any gesture or audio logic

### Offline ML (`ml/`)

- **scikit-learn** - MLP + RandomForest training, train/test split, evaluation
- **NumPy** - landmark normalization
- **matplotlib** - confusion matrix plot
- **joblib** - model persistence (reproducibility; not the deployed artifact — see above)
- **MediaPipe (Python)** + **remotezip** - landmark extraction from HaGRID, offline only
- **pytest** - tests for the data pipeline

### Tooling

- **uv** - Python package management
- **GitHub Actions** - CI on every pull request (ruff, pytest, ESLint)

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **uv** (Python package manager)
- **Webcam**

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/anastber/air-keys.git
   cd air-keys
   ```

2. **Install all dependencies**

   ```bash
   make install
   ```

   This creates a Python virtualenv, installs frontend dependencies, and — via an npm `postinstall` hook — copies the MediaPipe WASM runtime out of `node_modules` and downloads the pretrained gesture recognizer model into `frontend/public/`. Both are gitignored (too large for git, trivially reproducible), so this step is required after a fresh clone.

### Development

```bash
make dev
```

Starts the frontend at http://localhost:3000 (where everything actually runs) and the backend stub at http://localhost:8000. `make help` lists all commands.

## Playing a melody

The 8 always-available gestures are a fixed one-octave scale: fist=do, point=re, peace=mi, open palm=fa, pinch=sol, OK sign=la, rock on=ti, call me=do (an octave up). Hold your hand at a steady height and cycle through the shapes in order — the Songs tab (in the deck) has ready-made sequences for Twinkle Twinkle Little Star, Ode to Joy, and Mary Had a Little Lamb, all playable within this single octave. Raising or lowering your hand shifts the octave instead of changing the note, and each gesture's octave range is still editable in the Rules tab if you want that back as a continuous pitch control.

## Teaching a gesture

1. Open the app, show a hand to the camera.
2. Name a pose in the "Teach a Gesture" panel and hit Record — it captures ~20 samples while you hold the pose.
3. It's immediately classified from then on, with a default rule assigned automatically. Edit what it plays in the Gesture Rules panel.

Taught gestures live only in your browser's `localStorage` — clearing site data removes them, and they don't transfer to another device or browser.

## Contributing

Solo project, but every change goes through a pull request:

- **One branch per task**, named by type: `feat/`, `fix/`, `refactor/`, `docs/`, `chore/`, `test/`, or `exp/` for ML experiments (e.g. `feat/hagrid-extraction`). Tasks map to items in [ROADMAP.md](ROADMAP.md).
- **Pull request into `main`**, squash-merged once CI (lint + tests) passes.
- **Commit messages**: short, imperative, in English (`Add subject-disjoint split`, not `added split`).
- **Tests** cover the ML pipeline (`tests/`); UI changes are still verified by running the app (`make dev`).

```bash
make lint     # ruff + ESLint
make test     # pytest
make format   # ruff format
```

## License

MIT License - see LICENSE file for details.

The HaGRID dataset used for training and evaluation ([Kapitanov et al.](https://arxiv.org/abs/2206.08219), [repository](https://github.com/hukenovs/hagrid)) is distributed under its own [license](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf), a variant of CC BY-SA 4.0; it is not redistributed here.

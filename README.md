# AirKeys

**Camera-Based Gesture Instrument**

AirKeys turns your webcam into a musical instrument. Eight gestures — fist, point, peace, open palm, pinch, OK sign, rock on, call me — each always play the same fixed note, laid out as a one-octave diatonic scale (do through do), so cycling through them plays an actual, reproducible melody instead of an approximate one (see the in-app Songs tab for ready-made tunes). Everything — hand tracking, gesture recognition, and audio — runs client-side, in your browser, with no account and no setup. You can also teach it a gesture of your own, trained live on your own hand, private to your browser; on top of that, three of the eight (OK sign, rock on, call me) are classified by a small neural net *I* trained offline on a dataset I collected myself (see [Self-trained gesture classifier](#self-trained-gesture-classifier) below).

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
- **Rule engine**: whichever gesture wins (your taught gestures take priority, then the self-trained model, then pinch, then the canned poses) is mapped to a musical action. The 8 always-available gestures each play a *fixed* scale degree (do through do — see the Songs tab for melodies that use only this range), so the same shape always sounds the same note; hand height is repurposed as an optional octave shift rather than continuous pitch, widen a gesture's octave range in the Rules tab to bring that back. Taught gestures keep the original continuous height-to-pitch behavior. This mapping is editable live in the UI, not fixed in code.
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

**Data**: a temporary dev tool (`/collect`, `DatasetCollector.tsx`) runs the exact same in-browser MediaPipe pipeline used at inference time and lets me record labeled landmark samples, exported as JSON. 2,120 samples total across the 4 classes, both hands, varying distance/rotation/position across ~15-19 separate recording bursts per class.

**Catching a leakage bug**: my first pass split samples into train/test *per frame*, and got 100% accuracy on both a RandomForest baseline and the MLP — a result to be suspicious of, not proud of. Each "recording" burst captures ~20 near-identical consecutive frames, so a per-frame random split put frames from the same burst on both sides of the split; the model was partly being tested on near-duplicates of its own training data. The fix (`ml/train.py::assign_recording_bursts` + `group_stratified_split`): group samples by recording burst (same label, no >2s gap) and hold out whole bursts, never splitting one across train and test.

**Real held-out results**, after the fix:

| Model | Held-out accuracy | Notes |
|---|---|---|
| RandomForest (baseline) | 96.6% | Confusion mainly between `ok_sign` and `no_gesture` |
| MLP, hidden layers (32, 16) — **deployed** | 99.8% | See `ml/experiments/confusion_matrix.png` |

Believable, not suspicious: these are 4 visually distinct static poses from one recording session, not in-the-wild gesture recognition — a harder, noisier problem this pipeline doesn't yet attempt.

**Feature engineering**: the same wrist-centered, hand-span-scaled, handedness-mirrored normalization already used by the client-side kNN (`lib/customGestures.ts::normalizeLandmarks`), ported to Python (`ml/features.py`) so train-time and inference-time preprocessing stay identical.

**Deployment, without an inference runtime**: the MLP is small enough that its weight matrices (`coefs_`/`intercepts_`) export straight to JSON, and `lib/trainedGestures.ts` hand-rolls the forward pass (two dense layers + ReLU + softmax) in ~30 lines of TypeScript — no ONNX Runtime Web or TensorFlow.js dependency, consistent with this app's existing precedent of hand-rolling classification client-side (the kNN does the same).

**Reproducing it**: `uv run python -m ml.train` (needs `ml/data/gestures_v1.json`, collected via `/collect`). The exported weights JSON and confusion matrix are committed to git as evidence + the deployable artifact; the raw dataset and full sklearn pickle aren't (privacy + size) — see `.gitignore`. `frontend/scripts/setup-mediapipe.mjs` copies the committed weights into the gitignored `public/` dir on every install.

## Features

- **Real-time hand tracking** — MediaPipe, both hands, fully client-side
- **Zero-setup base gestures** — pinch, fist, open palm, point, peace, thumbs up all work on page load
- **Self-trained gesture classifier** — OK sign, rock on, call me, classified by an MLP trained offline with a real held-out evaluation (96.6%/99.8% accuracy — see above)
- **Playable melodies** — the 8 always-available gestures form a fixed one-octave diatonic scale; the Songs tab has gesture sequences for a few famous tunes
- **Personal gestures** — teach a new pose in ~20 samples, classified by kNN, private to your browser
- **Rule-based gesture-to-music mapping** — what each gesture plays is editable live, not hardcoded

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

### Tooling

- **uv** - Python package management
- **GitHub Actions** - lint CI (ruff + ESLint)

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

Starts the frontend at http://localhost:3000 (where everything actually runs) and the backend stub at http://localhost:8000.

```bash
make lint      # ruff + ESLint
make format    # ruff format
make help      # all available commands
```

## Playing a melody

The 8 always-available gestures are a fixed one-octave scale: fist=do, point=re, peace=mi, open palm=fa, pinch=sol, OK sign=la, rock on=ti, call me=do (an octave up). Hold your hand at a steady height and cycle through the shapes in order — the Songs tab (in the deck) has ready-made sequences for Twinkle Twinkle Little Star, Ode to Joy, and Mary Had a Little Lamb, all playable within this single octave. Raising or lowering your hand shifts the octave instead of changing the note, and each gesture's octave range is still editable in the Rules tab if you want that back as a continuous pitch control.

## Teaching a gesture

1. Open the app, show a hand to the camera.
2. Name a pose in the "Teach a Gesture" panel and hit Record — it captures ~20 samples while you hold the pose.
3. It's immediately classified from then on, with a default rule assigned automatically. Edit what it plays in the Gesture Rules panel.

Taught gestures live only in your browser's `localStorage` — clearing site data removes them, and they don't transfer to another device or browser.

## Contributing

Solo project — commits go straight to `main`, no feature branches. Verify changes by running the app live (`make dev`) rather than automated tests.

```bash
make lint
```

## License

MIT License - see LICENSE file for details.

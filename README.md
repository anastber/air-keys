# AirKeys

**Camera-Based Gesture Instrument**

AirKeys turns your webcam into a musical instrument. MediaPipe tracks your hand in real time; a small classifier you train live recognizes hand poses (fist, open palm, pinch, ...); a rule engine maps each recognized pose to a musical action. Nothing is scripted with hardcoded geometry checks — the pose recognition is a model trained on samples you record yourself, and can be retaught on the spot.

## How it works

```
Webcam --> MediaPipe (21 hand landmarks) --> Gesture Classifier --> Rule Engine --> Tone.js audio
              [browser + backend]           [trained live,          [gesture ->
                                              random forest]          musical action]
```

- **Hand tracking**: MediaPipe extracts 21 3D landmarks per hand from webcam frames, streamed over a WebSocket to the backend.
- **Gesture classifier**: landmarks are normalized (centered on the wrist, scaled by hand size, mirrored for left hands) into a pose-invariant feature vector, then classified by a random forest trained on samples recorded through the app. You can add a new gesture and retrain in seconds — see `/api/gestures/*` below.
- **Rule engine**: each recognized gesture is mapped to a musical action (play a note, play a chord, trigger an arpeggio, ...), with continuous signals like hand height and velocity controlling pitch and dynamics. This mapping is user-editable, not fixed.
- **Audio**: Tone.js synthesizes the result client-side.

## Features

- **Real-time hand tracking** — MediaPipe, 21-point landmarks, both hands
- **Trainable gesture classifier** — record labeled samples live, retrain in place, no ML background required to add a new gesture
- **Rule-based gesture-to-music mapping** — what each gesture does is configurable, not hardcoded

## Tech Stack

### Backend

- **FastAPI** - API server + WebSocket landmark streaming
- **MediaPipe** - hand landmark detection
- **scikit-learn** - gesture pose classifier (random forest)
- **OpenCV / NumPy** - frame decoding and feature math

### Frontend

- **Next.js 14 / TypeScript** - web interface
- **TailwindCSS** - styling
- **Tone.js** - web audio synthesis

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

   This will:
   - Create a Python virtual environment with uv
   - Install all Python dependencies from pyproject.toml
   - Install Node.js dependencies for the frontend

### Development

**Start the development servers:**

```bash
make dev
```

This starts both:

- **Backend API** http://localhost:8000
- **Frontend Web App** http://localhost:3000

**Lint and format code:**

```bash
make lint
make format
```

**View all available commands:**

```bash
make help
```

## Training a gesture

1. Open the app and hold a pose (e.g. a fist) in front of the camera.
2. Record ~15-30 samples for it via the training UI (or directly: `POST /api/gestures/samples` with `{label, landmarks, handedness}`).
3. Repeat for at least one more gesture (at least 8 samples per class, 2 classes minimum).
4. `POST /api/gestures/train` — retrains the classifier on every sample recorded so far and returns accuracy plus a confusion matrix. Takes well under a second.
5. The `/ws/landmarks` stream now includes a `gesture` + `confidence` field per detected hand.

Gesture labels are fixed to a set of 6 (`backend/gesture/labels.py`): `fist`, `open_palm`, `pinch`, `point`, `peace`, `thumbs_up`. Recorded samples live in `ml/data/gestures.jsonl`; the trained model in `ml/experiments/gesture_classifier.joblib` — both generated locally and gitignored, not shipped in the repo.

## Development Workflow

**Git workflow:**

- Each session = one feature branch
- All CI checks must pass before merge
- Use conventional commits: `feat(gesture): add classifier training endpoint`
- Verify features by running the app live (`make dev`) rather than automated tests

## Roadmap

- Rule-editor UI to remap gestures to musical actions without touching code
- Natural-language instrument reconfiguration (e.g. "make it sound sad and jazzy") via an LLM producing structured scale/voicing/envelope config
- Move MediaPipe inference into the browser to cut round-trip latency

## Contributing

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes and verify them live via `make dev`
3. Run `make lint`
4. Open a Pull Request with a clear description

## License

MIT License - see LICENSE file for details.

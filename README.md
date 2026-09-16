# AirKeys

**Camera-Based Gesture Instrument**

AirKeys turns your webcam into a musical instrument. Pinch, make a fist, or hold up a peace sign and it plays; hand height sets pitch. Everything — hand tracking, gesture recognition, and audio — runs client-side, in your browser, with no account and no setup. You can also teach it a gesture of your own, trained live on your own hand, private to your browser.

## How it works

```
Webcam --> MediaPipe (21 landmarks + pretrained gesture)  ---\
       --> pinch (pure geometry)                            >--> Rule Engine --> Tone.js audio
       --> your taught gestures (client-side kNN)          -/
                    [all in the browser]                    [gesture -> musical action]
```

- **Hand tracking + base gestures**: MediaPipe's pretrained `GestureRecognizer` extracts 21 hand landmarks and classifies 5 canned poses (fist, open palm, point, peace, thumbs up) in one model call, running fully client-side via WASM. It was trained by Google on a large, diverse hand dataset, so it works for any visitor immediately — no training step, no per-user setup.
- **Pinch**: not a model at all — a plain thumb-to-index-fingertip distance check. Geometry generalizes to any hand by construction, so this one didn't need ML.
- **Personalization**: teach it a gesture of your own and it's classified by k-nearest-neighbors against samples you record, right there in the training panel. No training step beyond recording — a sample is added, that's it. Everything stays in your browser's `localStorage`; nothing is sent to a server.
- **Rule engine**: whichever gesture wins (your taught gestures take priority, then pinch, then the canned poses) is mapped to a musical action — a note, a chord, an arpeggio, a bass note, a sustain toggle — with hand height picking pitch within an octave range. This mapping is editable live in the UI, not fixed in code.
- **Audio**: Tone.js synthesizes the result.

## Why it's built this way — a case study in getting this wrong first

The first version of gesture recognition here trained a random forest classifier **on the backend**, live, per visit: a visitor would record samples through the UI, hit train, and the server would fit a new model on the spot. It worked well as a single-person demo — real train/held-out-test evaluation, a confusion matrix, sub-second retraining.

It breaks the moment more than one person can use the app at once. The backend held **one shared model and one shared dataset file** for every visitor. A second visitor training their own gesture would silently retrain — and replace — the model the first visitor was mid-session with. There was no concept of "user" anywhere in that design. Fine for a personal demo running on one laptop; not something you can point a public URL at.

The fix wasn't more infrastructure (per-session backend isolation, cleanup jobs, etc.) — it was recognizing that most of what needs to be personalized doesn't need a server at all:

- **Base gestures** moved to a pretrained model (MediaPipe's `GestureRecognizer`) instead of something trained per-user, so there's no cold start and nothing to keep consistent across visitors.
- **Personalization** moved into the browser (`localStorage` + a from-scratch kNN classifier), so each visitor's taught gestures are private to them by construction — there's no shared state to corrupt because there's no shared state.
- As a side effect, the backend round-trip (WebSocket video streaming to a Python MediaPipe process) disappeared too, along with the latency it added — landmark extraction now happens directly against the `<video>` element.

The backend is now a stub, kept for a planned LLM endpoint (see Roadmap) rather than doing any gesture work at all.

## Features

- **Real-time hand tracking** — MediaPipe, both hands, fully client-side
- **Zero-setup base gestures** — pinch, fist, open palm, point, peace, thumbs up all work on page load
- **Personal gestures** — teach a new pose in ~20 samples, classified by kNN, private to your browser
- **Rule-based gesture-to-music mapping** — what each gesture plays is editable live, not hardcoded

## Tech Stack

### Frontend

- **Next.js 14 / TypeScript** - web interface
- **@mediapipe/tasks-vision** - pretrained hand landmark + gesture recognition, running in-browser via WASM
- **Tone.js** - web audio synthesis
- **TailwindCSS** - styling

### Backend

- **FastAPI** - currently a stub; reserved for the LLM reconfiguration layer (see Roadmap)

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

## Teaching a gesture

1. Open the app, show a hand to the camera.
2. Name a pose in the "Teach a Gesture" panel and hit Record — it captures ~20 samples while you hold the pose.
3. It's immediately classified from then on, with a default rule assigned automatically. Edit what it plays in the Gesture Rules panel.

Taught gestures live only in your browser's `localStorage` — clearing site data removes them, and they don't transfer to another device or browser.

## Roadmap

- Natural-language instrument reconfiguration (e.g. "make it sound sad and jazzy") via an LLM producing structured scale/voicing/envelope config — the next planned use for the backend stub
- A RAG-based music theory coach that grounds its explanations in what you just played
- Optional: an opt-in, anonymized pipeline that periodically improves the base model offline from aggregated taught-gesture data across visitors — a real continuous-learning story, without reintroducing a live shared model

## Contributing

Solo project — commits go straight to `main`, no feature branches. Verify changes by running the app live (`make dev`) rather than automated tests.

```bash
make lint
```

## License

MIT License - see LICENSE file for details.

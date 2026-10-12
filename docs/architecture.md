# Architecture

Design choices behind AirKeys. For the ML methodology and results, see [ROADMAP.md](../ROADMAP.md) and [ml/experiments/README.md](../ml/experiments/README.md).

## Everything runs in the browser

The first version trained a random forest **on the backend**, live: a visitor recorded samples, hit train, and the server fit a model on the spot. It worked as a single-person demo, and broke as soon as two people used it: the backend held **one shared model and one shared dataset** for everyone, so a second visitor training a gesture silently replaced the first visitor's model.

The fix wasn't more infrastructure (per-session isolation, cleanup jobs) but noticing that none of this needed a server:

- **Base gestures** come from a pretrained model (MediaPipe's `GestureRecognizer`), so there is no cold start and nothing to keep consistent across visitors.
- **Personalization** lives in the browser (`localStorage` + a from-scratch k-NN), so each visitor's gestures are private by construction — there is no shared state to corrupt.
- The WebSocket round-trip that streamed video to a Python MediaPipe process disappeared too, along with its latency: landmarks are now extracted directly from the `<video>` element.

The FastAPI backend is a stub, kept for future use.

## Gesture layers

Four recognizers run on every frame; the first one to fire wins:

1. **Taught gestures** — k-NN over the visitor's own samples (`lib/customGestures.ts`).
2. **Self-trained MLP** — OK sign, rock on, call me (`lib/trainedGestures.ts`).
3. **Pinch** — thumb-to-index distance; geometry generalizes to any hand, so no model is needed.
4. **MediaPipe's canned poses** — fist, open palm, point, peace, thumbs up (`lib/gestureRecognition.ts`).

The winning gesture goes through a rule engine that maps it to a musical action, editable live in the UI (`lib/rules.ts`).

## The self-trained classifier

The MLP (hidden layers 32 and 16) was trained on 2,120 landmark samples of my own hand, recorded with the `/collect` dev page, which runs the exact same MediaPipe pipeline as the app. Whole recording bursts were held out for testing, so near-identical consecutive frames never sat on both sides of the split: 99.8 % accuracy (random forest baseline: 96.6 %). Being single-subject, that number doesn't carry over to other hands — the re-evaluation is in the [README](../README.md#gesture-recognition-where-it-stands).

**Features.** Landmarks are centered on the wrist, scaled by hand span and mirrored for left hands, so a pose looks the same wherever it is in the frame. The normalization exists in Python (`ml/features.py`, training) and TypeScript (`lib/customGestures.ts`, inference).

**Shipping without an inference runtime.** The model is small enough that its weight matrices export straight to JSON, and the browser runs the forward pass in ~30 lines of TypeScript — no ONNX Runtime or TensorFlow.js dependency.

**Parity.** Since models are trained in Python but run in TypeScript, a shared fixture checks that both compute the same features and the same probabilities (`tests/test_parity.py`, `frontend/src/lib/parity.test.ts`), in CI.

## Offline ML pipeline (`ml/`)

| Module | Role |
|---|---|
| `datasets/hagrid.py` | Extracts landmarks from a sample of HaGRID, reading images straight out of the remote archives |
| `validation.py` | Dataset checks: schema, subject leakage, balance, duplicates (`make validate`) |
| `models.py`, `protocols.py` | Candidate models, the shipped MLP, evaluation protocols |
| `eval.py`, `summarize.py` | One evaluation run per command (`make eval`), and the results summary |

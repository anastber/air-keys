# AirKeys

**A camera-based gesture instrument that runs entirely in the browser** — [live demo](https://air-keys-delta.vercel.app/).

Eight hand gestures each play a fixed note of a one-octave scale, so you can play real melodies (the in-app Songs tab has a few). You can also teach it gestures of your own. Hand tracking, gesture recognition and audio all run client-side: no account, no setup, nothing sent to a server.

The project is also a testbed for applied ML done properly — evaluating a gesture classifier on people it has never seen, then fixing what that reveals. It is a work in progress: see [ROADMAP.md](ROADMAP.md).

## How it works

```
Webcam --> MediaPipe (21 landmarks + pretrained gesture)  ---\
       --> pinch (pure geometry)                            |
       --> self-trained MLP (offline-trained, 3 poses)       >--> Rule Engine --> Tone.js audio
       --> your taught gestures (client-side kNN)          -/
                    [all in the browser]                    [gesture -> musical action]
```

- **MediaPipe** extracts 21 hand landmarks and recognizes 5 base poses (fist, open palm, point, peace, thumbs up).
- **Pinch** is plain geometry: the thumb-to-index distance.
- A **self-trained MLP** recognizes OK sign, rock on and call me.
- **Your own gestures** are classified by a from-scratch k-NN, stored in your browser only.
- A **rule engine** maps the winning gesture to a note (editable live), and **Tone.js** plays it.

Built with Next.js, TypeScript, MediaPipe, Tone.js and scikit-learn. Design choices — why everything runs client-side, how the models ship without an inference runtime: [docs/architecture.md](docs/architecture.md).

## Gesture recognition: where it stands

The shipped MLP was trained on my hand only. Re-evaluated on other people's hands ([HaGRID](https://github.com/hukenovs/hagrid), 9k subjects) and on live webcam recordings:

| Evaluation | Accuracy |
|---|---|
| Shipped MLP — my hand (original test) | 99.8 % |
| Shipped MLP — 1,309 people it never saw | **71.9 %** [70.2, 73.6] |
| Logistic regression trained on 6,753 HaGRID people, 9 gestures — people it never saw | **98.0 %** [97.4, 98.5] |
| Same model — my live webcam, the 8 gestures | 99.0 % |
| Same model — my live webcam, recognizing "no gesture" | **12.5 %** |

Brackets are bootstrap 95 % confidence intervals over subjects. Generalizing to new hands is solved by training on many people. The open problem is **rejection**: resting or half-formed hands are taken for gestures, which would play wrong notes. That is the next phase of the roadmap; the app keeps the original MLP until then.

More detail:
- [ROADMAP.md](ROADMAP.md) — methodology, plan and progress
- [ml/experiments/README.md](ml/experiments/README.md) — every evaluation run, and how to reproduce it
- [docs/datasheet.md](docs/datasheet.md) — the datasets: HaGRID landmarks and my own webcam recordings

## Quick start

Requires Python 3.11+, Node.js 18+, [uv](https://docs.astral.sh/uv/) and a webcam.

```bash
git clone https://github.com/anastber/air-keys.git && cd air-keys
make install   # Python + Node dependencies; downloads the MediaPipe model into frontend/public/
make dev       # http://localhost:3000
```

`make help` lists the other commands: tests, lint, dataset validation, model evaluation.

## Playing

The 8 gestures form one octave: fist = do, point = re, peace = mi, open palm = fa, pinch = sol, OK sign = la, rock on = ti, call me = do (an octave up). Raising or lowering your hand shifts the octave. The Songs tab has ready-made melodies, and the Rules tab remaps any gesture.

To add a gesture of your own, name it in the "Teach a Gesture" panel and hit Record: it captures ~20 samples and is recognized right away. Taught gestures stay in your browser's `localStorage`.

## License

MIT — see [LICENSE](LICENSE). The HaGRID dataset used for training and evaluation ([Kapitanov et al.](https://arxiv.org/abs/2206.08219)) is distributed under its own [license](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf), a variant of CC BY-SA 4.0, and is not redistributed here.

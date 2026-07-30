# AirKeys

**Camera-Based Gesture Instrument**

AirKeys turns your MacBook webcam into a musical instrument. Point the camera at your hand, and a fine-tuned gesture classifier maps your finger positions and movements to piano or guitar notes in real time. A RAG-powered music theory assistant watches you play and offers contextual coaching, grounded in open music theory literature.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe for 21-point hand landmark detection
- **ML-Powered Gesture Recognition**: Fine-tuned transformer classifier with LoRA
- **Live Audio Synthesis**: MIDI-based note generation with piano and guitar sounds
- **Agentic RAG Assistant**: AI music theory coach powered by LangGraph and ChromaDB
- **Full MLOps Pipeline**: DVC data versioning, MLflow tracking, automated retraining

## Tech Stack

### Core ML & AI

- **PyTorch** - Deep learning framework
- **Transformers + LoRA** - Fine-tuned gesture classifier
- **MediaPipe** - Computer vision and hand tracking
- **sentence-transformers** - Vector embeddings for RAG

### Backend & Infrastructure

- **FastAPI** - High-performance API server
- **ChromaDB** - Vector database for music theory knowledge
- **LangGraph** - Agentic RAG workflow orchestration
- **MLflow** - Experiment tracking and model registry

### Frontend

- **Next.js 14** - React-based web interface
- **TypeScript** - Type-safe development
- **TailwindCSS** - Utility-first styling
- **Tone.js** - Web audio synthesis

### MLOps & Data

- **DVC** - Data version control
- **GitHub Actions** - CI/CD pipeline
- **uv** - Fast Python package management

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **uv** (Python package manager)
- **Webcam** (built-in MacBook camera works great)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/anastber/air-keys.git
   cd airkeys
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

## Development Workflow

**Git workflow:**

- Each session = one feature branch
- All CI checks must pass before merge
- Use conventional commits: `feat(gesture): add transformer classifier`
- Verify features by running the app live (`make dev`) rather than automated tests

## MLOps Pipeline

- **Data Versioning**: DVC tracks datasets and model artifacts
- **Experiment Tracking**: MLflow logs metrics, hyperparameters, models
- **Model Registry**: Automated promotion staging � production
- **CI/CD**: GitHub Actions for linting
- **Auto-Retraining**: Triggered when new training data is added

## Contributing

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes and verify them live via `make dev`
3. Run `make lint`
4. Open a Pull Request with a clear description

## License

MIT License - see LICENSE file for details.

---

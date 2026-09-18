// A genuinely self-trained gesture classifier: an MLP trained offline in
// Python (see ml/train.py) on a labeled dataset collected through
// app/collect/page.tsx, with a real train/test split and held-out
// evaluation (README has the numbers). Unlike lib/customGestures.ts's kNN,
// this model is fit once by the developer and ships with the app — a third
// gesture tier, for poses MediaPipe's pretrained recognizer doesn't cover.
//
// Deployed as a hand-rolled forward pass over the exported weight matrices
// rather than via ONNX/TF.js: the model is tiny (two hidden layers), so a
// ~30-line matrix multiply avoids pulling in an inference runtime, matching
// this app's existing precedent of hand-rolling classification client-side
// (see customGestures.ts's from-scratch kNN).

import { normalizeLandmarks } from '@/lib/customGestures';
import type { Landmark } from '@/lib/types';

// Never becomes an actionable label — exists purely so the model has
// somewhere to put "hand is just resting / mid-transition" instead of being
// forced to pick one of the real classes every frame.
const BACKGROUND_LABEL = 'no_gesture';

interface MlpLayer {
  weights: number[][]; // [inputDim][outputDim]
  biases: number[]; // [outputDim]
}

interface MlpWeights {
  classLabels: string[];
  confidenceThreshold: number;
  layers: MlpLayer[];
}

let modelPromise: Promise<MlpWeights> | null = null;
let loadedModel: MlpWeights | null = null;

/** Loads (once, memoized) the trained MLP's weights from the static asset ml/train.py exports. */
export function loadTrainedGestureModel(): Promise<MlpWeights> {
  if (!modelPromise) {
    modelPromise = fetch('/models/gesture_mlp_weights.json')
      .then((res) => res.json() as Promise<MlpWeights>)
      .then((model) => {
        loadedModel = model;
        return model;
      });
  }
  return modelPromise;
}

function relu(v: number[]): number[] {
  return v.map((x) => Math.max(0, x));
}

function softmax(v: number[]): number[] {
  const max = Math.max(...v);
  const exps = v.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

// One dense layer: y = x @ W + b, where W is [inputDim][outputDim].
function denseLayer(x: number[], layer: MlpLayer): number[] {
  const outputDim = layer.biases.length;
  const y = new Array(outputDim).fill(0);
  for (let i = 0; i < x.length; i++) {
    const row = layer.weights[i];
    for (let j = 0; j < outputDim; j++) {
      y[j] += x[i] * row[j];
    }
  }
  for (let j = 0; j < outputDim; j++) y[j] += layer.biases[j];
  return y;
}

function forward(features: number[], model: MlpWeights): number[] {
  let activations = features;
  for (let i = 0; i < model.layers.length; i++) {
    const raw = denseLayer(activations, model.layers[i]);
    activations = i < model.layers.length - 1 ? relu(raw) : raw;
  }
  return softmax(activations);
}

/**
 * Classify a hand against the self-trained model. Returns null if the
 * model hasn't finished loading yet, the winning class is the background
 * class, or its confidence is below the trained threshold.
 */
export function predictTrainedGesture(
  landmarks: Landmark[],
  handedness: string
): { label: string; confidence: number } | null {
  if (!loadedModel) return null;

  const features = normalizeLandmarks(landmarks, handedness);
  const probs = forward(features, loadedModel);

  let bestIndex = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIndex]) bestIndex = i;
  }

  const label = loadedModel.classLabels[bestIndex];
  const confidence = probs[bestIndex];
  if (label === BACKGROUND_LABEL || confidence < loadedModel.confidenceThreshold) return null;

  return { label, confidence };
}

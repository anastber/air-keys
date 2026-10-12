// Per-frame inference cost of each candidate model, in V8 (Chrome's JS
// engine, here under Node) — Phase 0.5 of ROADMAP.md. Run: npm run bench.
//
// Uses the app's real code paths (normalizeLandmarks, the MLP forward pass)
// with weights of each candidate's shape; values don't affect timing.
// Random forest isn't benchmarked: the app has no tree-inference code.

import { bench, describe } from 'vitest';
import { normalizeLandmarks } from '@/lib/customGestures';
import { forward, type MlpLayer, type MlpWeights } from '@/lib/trainedGestures';
import type { Landmark } from '@/lib/types';

const FEATURES = 63;
const CLASSES = 9;
// hagrid_landmarks_v1 training split size: what a kNN would have to scan.
const KNN_TRAINING_SAMPLES = 8362;

let seed = 1;
function random(): number {
  seed = (seed * 16807) % 2147483647;
  return seed / 2147483647;
}

function layer(inputs: number, outputs: number): MlpLayer {
  return {
    weights: Array.from({ length: inputs }, () => Array.from({ length: outputs }, random)),
    biases: Array.from({ length: outputs }, random),
  };
}

function model(sizes: number[]): MlpWeights {
  return {
    classLabels: [],
    confidenceThreshold: 0,
    layers: sizes.slice(1).map((size, i) => layer(sizes[i], size)),
  };
}

const hand: Landmark[] = Array.from({ length: 21 }, () => ({ x: random(), y: random(), z: random() }));
const features = normalizeLandmarks(hand, 'Right');
const mlp = model([FEATURES, 32, 16, CLASSES]);
const logreg = model([FEATURES, CLASSES]); // a linear layer + softmax
const knnSamples = Array.from({ length: KNN_TRAINING_SAMPLES }, () =>
  Array.from({ length: FEATURES }, random)
);

function nearestDistances(x: number[]): number[] {
  return knnSamples.map((sample) => {
    let sum = 0;
    for (let i = 0; i < FEATURES; i++) {
      const d = x[i] - sample[i];
      sum += d * d;
    }
    return sum;
  });
}

describe('per-frame inference', () => {
  bench('normalize landmarks', () => {
    normalizeLandmarks(hand, 'Right');
  });
  bench('logreg (63 -> 9)', () => {
    forward(features, logreg);
  });
  bench('mlp (63 -> 32 -> 16 -> 9)', () => {
    forward(features, mlp);
  });
  bench(`knn (brute force over ${KNN_TRAINING_SAMPLES} samples)`, () => {
    nearestDistances(features);
  });
});

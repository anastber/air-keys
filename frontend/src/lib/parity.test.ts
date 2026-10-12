// Python <-> TypeScript parity: models are trained and evaluated in Python
// (ml/) but run here in the browser, so both sides must compute the same
// numbers. tests/fixtures/parity.json holds inputs and the outputs Python
// produces (see tests/parity_fixture.py); this checks the TypeScript side.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { normalizeLandmarks } from '@/lib/customGestures';
import { forward, type MlpWeights } from '@/lib/trainedGestures';
import type { Landmark } from '@/lib/types';

interface ParityCase {
  handedness: string;
  landmarks: Landmark[];
  features: number[];
  probabilities: number[];
}

const repoFile = (path: string) =>
  readFileSync(fileURLToPath(new URL(`../../../${path}`, import.meta.url)));

const fixture = JSON.parse(repoFile('tests/fixtures/parity.json').toString()) as {
  deployed_weights_sha256: string;
  cases: ParityCase[];
};
const weightsFile = repoFile('ml/experiments/gesture_mlp_weights.json');
const weights = JSON.parse(weightsFile.toString()) as MlpWeights;

// Python normalizes in float32, TypeScript in float64: compare relative to
// magnitude (features reach ~1e5 when the hand-span scale hits its floor).
const FEATURE_TOLERANCE = 1e-5;
const PROBABILITY_TOLERANCE = 1e-9;

function expectClose(actual: number[], expected: number[], tolerance: number) {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, i) => {
    const allowed = tolerance * Math.max(1, Math.abs(expected[i]));
    expect(Math.abs(value - expected[i]), `index ${i}`).toBeLessThanOrEqual(allowed);
  });
}

describe('Python/TypeScript parity', () => {
  it('uses the same deployed weights as the fixture', () => {
    const sha = createHash('sha256').update(weightsFile).digest('hex');
    expect(sha, 'regenerate with `uv run python -m tests.parity_fixture`').toBe(
      fixture.deployed_weights_sha256
    );
  });

  it.each(fixture.cases.map((c, i) => [i, c] as const))(
    'normalizes landmarks like ml/features.py (case %i)',
    (_, c) => {
      expectClose(normalizeLandmarks(c.landmarks, c.handedness), c.features, FEATURE_TOLERANCE);
    }
  );

  it.each(fixture.cases.map((c, i) => [i, c] as const))(
    'runs the deployed MLP like ml/models.py (case %i)',
    (_, c) => {
      // Fed Python's features, so this isolates the forward pass itself.
      expectClose(forward(c.features, weights), c.probabilities, PROBABILITY_TOLERANCE);
    }
  );
});

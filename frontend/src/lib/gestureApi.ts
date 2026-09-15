// Client for the backend's /api/gestures/* endpoints: recording labeled
// samples, training the classifier, and reading its current status.

import type { Landmark } from '@/lib/types';

const API_BASE = 'http://localhost:8000';

export interface GestureStatus {
  labels: string[];
  sample_counts: Record<string, number>;
  model_trained: boolean;
  trained_at: number | null;
  accuracy: number | null;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
}

export interface TrainResult {
  accuracy: number;
  held_out_evaluation: boolean;
  sample_counts: Record<string, number>;
  confusion_matrix: ConfusionMatrix;
  trained_at: number;
}

async function extractError(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    return body.detail ?? fallback;
  } catch {
    return fallback;
  }
}

export async function fetchGestureStatus(): Promise<GestureStatus> {
  const res = await fetch(`${API_BASE}/api/gestures/status`);
  if (!res.ok) {
    throw new Error(await extractError(res, 'Failed to fetch gesture status'));
  }
  return res.json();
}

export async function recordGestureSample(
  label: string,
  landmarks: Landmark[],
  handedness: string
): Promise<{ sample_counts: Record<string, number> }> {
  const res = await fetch(`${API_BASE}/api/gestures/samples`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, landmarks, handedness }),
  });
  if (!res.ok) {
    throw new Error(await extractError(res, 'Failed to record sample'));
  }
  return res.json();
}

export async function trainGestureClassifier(): Promise<TrainResult> {
  const res = await fetch(`${API_BASE}/api/gestures/train`, { method: 'POST' });
  if (!res.ok) {
    throw new Error(await extractError(res, 'Training failed'));
  }
  return res.json();
}

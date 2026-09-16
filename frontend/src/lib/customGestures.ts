// Client-side personalization layer: lets a visitor teach the instrument a
// gesture of their own, on top of the pretrained base set from
// lib/gestureRecognition.ts.
//
// Deliberately k-nearest-neighbors rather than anything trained: kNN has no
// training step at all — "recording a sample" IS the entire update, there's
// no fit() to run, no waiting. Everything (samples + lookup) stays in this
// browser's localStorage. Nothing is sent to a server: no shared model to
// corrupt across visitors, no per-user backend state to manage, complete
// privacy for the visitor's hand data. The normalization below is the same
// technique the backend's original classifier used (see git history) —
// wrist-center, scale by hand span, mirror left hands — ported to run here
// instead of server-side now that classification is fully client-side.

import type { Landmark } from '@/lib/types';

const WRIST = 0;
const MIDDLE_FINGER_MCP = 9;

const STORAGE_KEY = 'airkeys.customGestures.v1';
const K_NEIGHBORS = 5;
// Starting point, not empirically tuned — a normalized-feature Euclidean
// distance below this counts as "the same pose". Tighten it if gestures
// misfire, loosen it if a taught gesture won't trigger reliably.
const MAX_AVG_DISTANCE = 1.2;

interface CustomSample {
  label: string;
  features: number[];
}

/** Translation/scale/mirror-invariant feature vector — see backend/gesture/features.py for the original derivation. */
export function normalizeLandmarks(landmarks: Landmark[], handedness: string): number[] {
  const origin = landmarks[WRIST];
  const centered = landmarks.map((p) => ({
    x: p.x - origin.x,
    y: p.y - origin.y,
    z: p.z - origin.z,
  }));

  const ref = centered[MIDDLE_FINGER_MCP];
  const scale = Math.max(Math.hypot(ref.x, ref.y), 1e-6);
  const mirror = handedness === 'Left' ? -1 : 1;

  const features: number[] = [];
  for (const p of centered) {
    features.push((mirror * p.x) / scale, p.y / scale, p.z / scale);
  }
  return features;
}

function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function loadSamples(): CustomSample[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as CustomSample[]) : [];
  } catch {
    return [];
  }
}

function saveSamples(samples: CustomSample[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
  } catch {
    // Storage full or unavailable (e.g. private browsing) — personalization
    // just won't persist this session. Not worth surfacing as an error.
  }
}

/** Record one labeled sample. This is the entire "training" step. */
export function recordCustomGesture(label: string, landmarks: Landmark[], handedness: string): void {
  const samples = loadSamples();
  samples.push({ label, features: normalizeLandmarks(landmarks, handedness) });
  saveSamples(samples);
}

export function clearCustomGesture(label: string): void {
  saveSamples(loadSamples().filter((s) => s.label !== label));
}

export function customGestureCounts(): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const s of loadSamples()) {
    counts[s.label] = (counts[s.label] ?? 0) + 1;
  }
  return counts;
}

export function customGestureLabels(): string[] {
  return Object.keys(customGestureCounts());
}

/**
 * k-nearest-neighbors lookup against this visitor's own recorded samples.
 * Returns null if nothing's been taught yet, or the nearest matches aren't
 * close enough to be confident.
 */
export function predictCustomGesture(
  landmarks: Landmark[],
  handedness: string
): { label: string; distance: number } | null {
  const samples = loadSamples();
  if (samples.length === 0) return null;

  const query = normalizeLandmarks(landmarks, handedness);
  const nearest = samples
    .map((s) => ({ label: s.label, distance: euclideanDistance(query, s.features) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, Math.min(K_NEIGHBORS, samples.length));

  const votes: Record<string, { count: number; totalDistance: number }> = {};
  for (const n of nearest) {
    const v = votes[n.label] ?? { count: 0, totalDistance: 0 };
    v.count += 1;
    v.totalDistance += n.distance;
    votes[n.label] = v;
  }

  const winner = Object.entries(votes).sort((a, b) => b[1].count - a[1].count)[0];
  if (!winner) return null;

  const [label, { count, totalDistance }] = winner;
  const avgDistance = totalDistance / count;
  return avgDistance <= MAX_AVG_DISTANCE ? { label, distance: avgDistance } : null;
}

// Wraps MediaPipe's pretrained GestureRecognizer: one model call, running
// entirely client-side (WASM, self-hosted assets — see
// scripts/setup-mediapipe.mjs), gives both the 21 hand landmarks AND a
// classification into 7 canned poses. Trained by Google on a large, diverse
// hand dataset, so it works for any visitor's hand with zero setup — no
// per-user training step, no server round-trip.
//
// This is the "base" gesture layer. lib/customGestures.ts is the opt-in
// layer on top, for a gesture a visitor teaches it themselves.

import { FilesetResolver, GestureRecognizer } from '@mediapipe/tasks-vision';
import type { HandData, Landmark } from '@/lib/types';

// Translates MediaPipe's canned category names to the internal gesture
// labels the rule engine already uses (see lib/rules.ts). Only poses that
// have a sensible default musical action are translated — everything else
// (Thumb_Down, ILoveYou, None, low-confidence) reads as "no gesture", same
// as if nothing were detected.
const CANNED_GESTURE_MAP: Record<string, string> = {
  Closed_Fist: 'fist',
  Open_Palm: 'open_palm',
  Pointing_Up: 'point',
  Victory: 'peace',
  Thumb_Up: 'thumbs_up',
};

const MIN_CANNED_CONFIDENCE = 0.5;

let recognizerPromise: Promise<GestureRecognizer> | null = null;

/** Loads (once, memoized) the recognizer using the self-hosted WASM runtime and model. */
export function loadGestureRecognizer(): Promise<GestureRecognizer> {
  if (!recognizerPromise) {
    recognizerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks('/mediapipe-wasm');
      return GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/models/gesture_recognizer.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
      });
    })();
  }
  return recognizerPromise;
}

function toLandmarks(raw: { x: number; y: number; z: number }[]): Landmark[] {
  return raw.map((p) => ({ x: p.x, y: p.y, z: p.z }));
}

/** Run recognition on one video frame, shaped like the HandData the rest of the app expects. */
export function recognizeFrame(
  recognizer: GestureRecognizer,
  video: HTMLVideoElement,
  timestampMs: number
): HandData[] {
  const result = recognizer.recognizeForVideo(video, timestampMs);

  return result.landmarks.map((landmarks, i) => {
    const handedness = result.handedness[i]?.[0]?.categoryName ?? 'Unknown';
    const topGesture = result.gestures[i]?.[0];
    const gesture =
      topGesture && topGesture.score >= MIN_CANNED_CONFIDENCE
        ? CANNED_GESTURE_MAP[topGesture.categoryName]
        : undefined;

    return {
      handedness,
      landmarks: toLandmarks(landmarks),
      gesture,
      confidence: gesture ? topGesture!.score : undefined,
    };
  });
}

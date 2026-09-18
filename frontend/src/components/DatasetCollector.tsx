'use client';

// Temporary dev tool for collecting the labeled dataset that ml/train.py
// trains on. Not part of the production UI (see app/collect/page.tsx) —
// deliberately separate from GestureTrainer.tsx, whose recordings go
// straight into localStorage for the client-side kNN. This one instead
// accumulates raw samples in memory across many labels/recording rounds and
// exports them as a single JSON file, using the exact same MediaPipe
// pipeline (WebcamLandmarks) as production inference, so the offline
// training data matches what the model will see at inference time.

import React, { useCallback, useEffect, useState } from 'react';
import WebcamLandmarks from '@/components/WebcamLandmarks';
import { gestureEmoji } from '@/lib/gestureIcons';
import type { HandData, Landmark, LandmarkData } from '@/lib/types';

const SAMPLES_PER_RECORDING = 20;
const TARGET_SAMPLES_PER_CLASS = 250;

// The 4 classes ml/train.py expects (see lib/rules.ts::TRAINED_GESTURE_LABELS
// plus the no_gesture background class) — shown here so recording sessions
// don't drift from what the model will actually be trained on.
const TARGET_CLASSES: { label: string; description: string }[] = [
  { label: 'ok_sign', description: 'Thumb + index touching in a circle, other 3 fingers up' },
  { label: 'rock_on', description: 'Index + pinky extended, thumb/middle/ring folded' },
  { label: 'call_me', description: 'Thumb + pinky extended, other 3 fingers folded' },
  {
    label: 'no_gesture',
    description: 'Relaxed hand, resting pose, or mid-transition — vary this one the most',
  },
];

interface DatasetRecord {
  label: string;
  landmarks: Landmark[];
  handedness: string;
  timestamp: number;
}

const DatasetCollector: React.FC = () => {
  const [labelInput, setLabelInput] = useState('');
  const [recordingLabel, setRecordingLabel] = useState<string | null>(null);
  const [recordedCount, setRecordedCount] = useState(0);
  const [samples, setSamples] = useState<DatasetRecord[]>([]);
  const [currentHand, setCurrentHand] = useState<HandData | null>(null);

  const handleLandmarks = useCallback((data: LandmarkData) => {
    setCurrentHand(data.hands[0] ?? null);
  }, []);

  // While recording, capture one sample per incoming frame until the target
  // count is hit — same shape as GestureTrainer.tsx's capture effect, just
  // appending to in-memory state instead of writing to localStorage.
  useEffect(() => {
    if (!recordingLabel || !currentHand || currentHand.landmarks.length !== 21) return;

    if (recordedCount >= SAMPLES_PER_RECORDING) {
      setRecordingLabel(null);
      return;
    }

    setSamples((prev) => [
      ...prev,
      {
        label: recordingLabel,
        landmarks: currentHand.landmarks,
        handedness: currentHand.handedness,
        timestamp: Date.now() / 1000,
      },
    ]);
    setRecordedCount((c) => c + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentHand, recordingLabel, recordedCount]);

  const startRecording = () => {
    const label = labelInput.trim();
    if (!label) return;
    setRecordedCount(0);
    setRecordingLabel(label);
  };

  const clearLabel = (label: string) => {
    setSamples((prev) => prev.filter((s) => s.label !== label));
  };

  const downloadDataset = () => {
    const blob = new Blob([JSON.stringify(samples)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gestures_v1.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const counts: Record<string, number> = {};
  for (const s of samples) counts[s.label] = (counts[s.label] ?? 0) + 1;

  const progress = recordingLabel ? recordedCount / SAMPLES_PER_RECORDING : 0;

  return (
    <main className="min-h-screen bg-ak-bg text-ak-text p-6 flex flex-col gap-6 max-w-2xl mx-auto">
      <div>
        <h1 className="text-lg font-semibold">Dataset collector (dev tool)</h1>
        <p className="text-sm text-ak-muted mt-1">
          Train the 4 classes below, then save the export to{' '}
          <code>ml/data/gestures_v1.json</code>.
        </p>
      </div>

      {/* Target classes — click one to fill the label input below. */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {TARGET_CLASSES.map(({ label, description }) => {
          const count = samples.filter((s) => s.label === label).length;
          const done = count >= TARGET_SAMPLES_PER_CLASS;
          return (
            <button
              key={label}
              onClick={() => setLabelInput(label)}
              disabled={recordingLabel !== null}
              className={`text-left flex flex-col gap-1 rounded-xl border px-3 py-2.5 transition-colors disabled:opacity-50 ${
                labelInput === label
                  ? 'border-ak-violet/60 bg-ak-violet/10'
                  : 'border-white/10 bg-white/[0.03] hover:bg-white/[0.06]'
              }`}
            >
              <span className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 font-mono text-sm">
                  <span className="text-lg leading-none">{gestureEmoji(label)}</span>
                  {label}
                </span>
                <span className={`text-xs font-mono ${done ? 'text-ak-emerald' : 'text-ak-subtle'}`}>
                  {count}/{TARGET_SAMPLES_PER_CLASS}
                </span>
              </span>
              <span className="text-xs text-ak-muted">{description}</span>
            </button>
          );
        })}
      </div>

      {/* How-to, kept short — the counters above are the real feedback loop. */}
      <ol className="text-xs text-ak-muted list-decimal list-inside flex flex-col gap-1 bg-white/[0.03] border border-white/[0.06] rounded-xl px-4 py-3">
        <li>Click a class above to fill its label, then hit Record and hold the pose.</li>
        <li>Each Record captures ~1s (20 frames) — repeat 10-15x per class, not once.</li>
        <li>
          Between recordings, change distance, rotation, hand position in frame, and
          which hand — a single long hold gives near-identical frames.
        </li>
        <li>
          <code>no_gesture</code> needs the most variety: open/relaxed/curled hand, mid-transition
          poses, different distances — this is what stops false triggers at rest.
        </li>
        <li>Download the dataset once every class shows ~250+ samples.</li>
      </ol>

      <div className="rounded-xl overflow-hidden border border-white/10">
        <WebcamLandmarks onLandmarks={handleLandmarks} />
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={labelInput}
          onChange={(e) => setLabelInput(e.target.value)}
          placeholder="e.g. ok_sign"
          disabled={recordingLabel !== null}
          className="flex-1 bg-white/[0.06] border border-white/10 rounded-lg px-3 py-1.5 text-sm placeholder:text-ak-subtle focus:outline-none focus:ring-1 focus:ring-ak-violet/60 disabled:opacity-50"
        />
        <button
          onClick={startRecording}
          disabled={!currentHand || !labelInput.trim() || recordingLabel !== null}
          className="px-4 py-1.5 rounded-lg text-sm font-medium bg-gradient-to-r from-ak-violet to-ak-cyan text-white disabled:opacity-40"
        >
          Record
        </button>
      </div>

      {recordingLabel && (
        <p className="text-sm font-mono text-ak-cyan">
          &ldquo;{recordingLabel}&rdquo; — {recordedCount}/{SAMPLES_PER_RECORDING} ({Math.round(progress * 100)}%)
        </p>
      )}
      {!currentHand && <p className="text-xs text-ak-amber">Show a hand to the camera to record.</p>}

      {Object.keys(counts).length > 0 && (
        <div className="flex flex-col gap-1.5 pt-2 border-t border-white/[0.06]">
          {Object.entries(counts).map(([label, count]) => (
            <div key={label} className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-1.5">
                <span>{gestureEmoji(label)}</span>
                <span className="font-mono">
                  {label} <span className="text-ak-subtle">({count})</span>
                </span>
              </span>
              <button
                onClick={() => clearLabel(label)}
                className="text-xs text-ak-red/80 hover:text-ak-red hover:underline"
              >
                remove
              </button>
            </div>
          ))}
          <button
            onClick={downloadDataset}
            className="mt-2 self-start px-4 py-1.5 rounded-lg text-sm font-medium bg-white/10 hover:bg-white/20"
          >
            Download dataset ({samples.length} samples)
          </button>
        </div>
      )}
    </main>
  );
};

export default DatasetCollector;

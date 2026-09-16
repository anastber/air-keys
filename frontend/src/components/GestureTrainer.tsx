'use client';

import React, { useEffect, useState } from 'react';
import {
  clearCustomGesture,
  customGestureCounts,
  recordCustomGesture,
} from '@/lib/customGestures';
import { gestureEmoji } from '@/lib/gestureIcons';
import type { HandData } from '@/lib/types';

interface GestureTrainerProps {
  // First detected hand this frame, or null when no hand is in view.
  currentHand: HandData | null;
  // Fires once a new gesture finishes recording, so the caller can give it
  // a default rule (the editor won't show a row for a gesture with no rule).
  onGestureRecorded: (label: string) => void;
}

const SAMPLES_PER_RECORDING = 20;

const GestureTrainer: React.FC<GestureTrainerProps> = ({ currentHand, onGestureRecorded }) => {
  const [labelInput, setLabelInput] = useState('');
  const [recordingLabel, setRecordingLabel] = useState<string | null>(null);
  const [recordedCount, setRecordedCount] = useState(0);
  const [counts, setCounts] = useState<Record<string, number>>({});

  useEffect(() => {
    setCounts(customGestureCounts());
  }, []);

  // While recording, capture one sample from every incoming frame until the
  // target count is hit. No network round-trip here (unlike an earlier,
  // server-trained version of this component) — recording a sample is a
  // synchronous localStorage write, so there's nothing to race against.
  useEffect(() => {
    if (!recordingLabel || !currentHand || currentHand.landmarks.length !== 21) return;

    if (recordedCount >= SAMPLES_PER_RECORDING) {
      setCounts(customGestureCounts());
      onGestureRecorded(recordingLabel);
      setRecordingLabel(null);
      return;
    }

    recordCustomGesture(recordingLabel, currentHand.landmarks, currentHand.handedness);
    setRecordedCount((c) => c + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentHand, recordingLabel, recordedCount]);

  const startRecording = () => {
    const label = labelInput.trim();
    if (!label) return;
    setRecordedCount(0);
    setRecordingLabel(label);
  };

  const handleDelete = (label: string) => {
    clearCustomGesture(label);
    setCounts(customGestureCounts());
  };

  const progress = recordingLabel ? recordedCount / SAMPLES_PER_RECORDING : 0;

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-ak-muted">
        Name a pose, hold it in frame, and record ~20 samples. Stored only in this
        browser — nothing is sent anywhere.
      </p>

      <div className="flex gap-2">
        <input
          type="text"
          value={labelInput}
          onChange={(e) => setLabelInput(e.target.value)}
          placeholder="e.g. rock_on"
          disabled={recordingLabel !== null}
          className="flex-1 bg-white/[0.06] border border-white/10 rounded-lg px-3 py-1.5 text-sm text-ak-text placeholder:text-ak-subtle focus:outline-none focus:ring-1 focus:ring-ak-violet/60 focus:border-ak-violet/60 disabled:opacity-50"
        />
        <button
          onClick={startRecording}
          disabled={!currentHand || !labelInput.trim() || recordingLabel !== null}
          className="px-4 py-1.5 rounded-lg text-sm font-medium bg-gradient-to-r from-ak-violet to-ak-cyan text-white shadow-lg shadow-ak-violet/20 hover:opacity-90 transition-opacity disabled:opacity-40 disabled:shadow-none"
        >
          Record
        </button>
      </div>

      {recordingLabel && (
        <div className="flex items-center gap-3">
          <RecordingRing progress={progress} />
          <p className="text-sm font-mono text-ak-cyan">
            &ldquo;{recordingLabel}&rdquo; — {recordedCount}/{SAMPLES_PER_RECORDING}
          </p>
        </div>
      )}
      {!currentHand && (
        <p className="text-xs text-ak-amber">Show a hand to the camera to record.</p>
      )}

      {Object.keys(counts).length > 0 && (
        <div className="flex flex-col gap-1.5 pt-1 border-t border-white/[0.06]">
          {Object.entries(counts).map(([label, count]) => (
            <div key={label} className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-1.5">
                <span className="text-base leading-none">{gestureEmoji(label)}</span>
                <span className="font-mono text-ak-text">
                  {label} <span className="text-ak-subtle">({count})</span>
                </span>
              </span>
              <button
                onClick={() => handleDelete(label)}
                className="text-xs text-ak-red/80 hover:text-ak-red hover:underline"
              >
                remove
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const RecordingRing: React.FC<{ progress: number }> = ({ progress }) => {
  const pct = Math.round(progress * 100);
  return (
    <div
      className="relative w-9 h-9 rounded-full shrink-0"
      style={{
        background: `conic-gradient(#22d3ee ${pct}%, rgba(255,255,255,0.1) ${pct}% 100%)`,
      }}
    >
      <div className="absolute inset-[3px] rounded-full bg-ak-bg" />
    </div>
  );
};

export default GestureTrainer;

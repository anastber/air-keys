'use client';

import React, { useEffect, useState } from 'react';
import {
  clearCustomGesture,
  customGestureCounts,
  recordCustomGesture,
} from '@/lib/customGestures';
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

  return (
    <div className="border rounded p-4 flex flex-col gap-3 w-full max-w-md">
      <h3 className="font-bold text-lg">Teach a Gesture</h3>
      <p className="text-sm text-gray-600">
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
          className="border rounded px-2 py-1 text-sm flex-1 disabled:opacity-50"
        />
        <button
          onClick={startRecording}
          disabled={!currentHand || !labelInput.trim() || recordingLabel !== null}
          className="px-3 py-1 rounded text-sm font-medium bg-blue-500 hover:bg-blue-600 text-white disabled:opacity-50"
        >
          Record
        </button>
      </div>

      {recordingLabel && (
        <p className="text-sm font-mono text-blue-600">
          Recording &ldquo;{recordingLabel}&rdquo;: {recordedCount}/{SAMPLES_PER_RECORDING}
        </p>
      )}
      {!currentHand && (
        <p className="text-xs text-amber-600">Show a hand to the camera to record.</p>
      )}

      {Object.keys(counts).length > 0 && (
        <div className="flex flex-col gap-1">
          {Object.entries(counts).map(([label, count]) => (
            <div key={label} className="flex items-center justify-between text-sm">
              <span className="font-mono">
                {label} ({count})
              </span>
              <button
                onClick={() => handleDelete(label)}
                className="text-xs text-red-600 hover:underline"
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

export default GestureTrainer;

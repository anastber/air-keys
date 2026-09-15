'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  fetchGestureStatus,
  recordGestureSample,
  trainGestureClassifier,
  type GestureStatus,
  type TrainResult,
} from '@/lib/gestureApi';
import type { HandData } from '@/lib/types';

interface GestureTrainerProps {
  // First detected hand this frame, or null when no hand is in view.
  currentHand: HandData | null;
  status: GestureStatus | null;
  onStatusChange: (status: GestureStatus) => void;
}

const SAMPLES_PER_RECORDING = 20;

const GestureTrainer: React.FC<GestureTrainerProps> = ({
  currentHand,
  status,
  onStatusChange,
}) => {
  const [recordingLabel, setRecordingLabel] = useState<string | null>(null);
  const [recordedCount, setRecordedCount] = useState(0);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const inFlightRef = useRef(false);

  const refreshStatus = async () => {
    try {
      onStatusChange(await fetchGestureStatus());
    } catch {
      // Best-effort refresh — leave the previously known status displayed.
    }
  };

  // While recording, capture one sample from every incoming frame for the
  // selected label until the target count is hit. Gated by inFlightRef the
  // same way frame sending is in WebcamLandmarks, so a slow request can't
  // overlap with the next one once frames arrive faster than the round-trip.
  useEffect(() => {
    if (!recordingLabel || !currentHand || inFlightRef.current) return;
    if (currentHand.landmarks.length !== 21) return;

    if (recordedCount >= SAMPLES_PER_RECORDING) {
      setRecordingLabel(null);
      refreshStatus();
      return;
    }

    inFlightRef.current = true;
    recordGestureSample(recordingLabel, currentHand.landmarks, currentHand.handedness)
      .then(() => setRecordedCount((c) => c + 1))
      .catch(() => setRecordingLabel(null))
      .finally(() => {
        inFlightRef.current = false;
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentHand, recordingLabel, recordedCount]);

  const startRecording = (label: string) => {
    setRecordedCount(0);
    setTrainResult(null);
    setTrainError(null);
    setRecordingLabel(label);
  };

  const handleTrain = async () => {
    setIsTraining(true);
    setTrainError(null);
    try {
      const result = await trainGestureClassifier();
      setTrainResult(result);
      await refreshStatus();
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      setIsTraining(false);
    }
  };

  if (!status) return null;

  return (
    <div className="border rounded p-4 flex flex-col gap-3 w-full max-w-md">
      <h3 className="font-bold text-lg">Train Gestures</h3>
      <p className="text-sm text-gray-600">
        Hold a pose in frame and record ~20 samples per gesture, then train. Needs
        at least 2 gestures with 8+ samples each.
      </p>

      <div className="grid grid-cols-2 gap-2">
        {status.labels.map((label) => {
          const count = status.sample_counts[label] ?? 0;
          const isActive = recordingLabel === label;
          return (
            <button
              key={label}
              onClick={() => startRecording(label)}
              disabled={!currentHand || (recordingLabel !== null && !isActive)}
              className={`px-3 py-2 rounded text-sm font-medium border disabled:opacity-50 ${
                isActive ? 'bg-blue-500 text-white' : 'bg-white hover:bg-gray-50'
              }`}
            >
              {label} ({count})
              {isActive ? ` — ${recordedCount}/${SAMPLES_PER_RECORDING}` : ''}
            </button>
          );
        })}
      </div>

      {!currentHand && (
        <p className="text-xs text-amber-600">Show a hand to the camera to record samples.</p>
      )}

      <button
        onClick={handleTrain}
        disabled={isTraining || recordingLabel !== null}
        className="px-4 py-2 rounded font-medium bg-green-600 hover:bg-green-700 text-white disabled:opacity-50"
      >
        {isTraining ? 'Training…' : 'Train Classifier'}
      </button>

      {trainError && <p className="text-sm text-red-600">{trainError}</p>}

      {trainResult && (
        <div className="text-sm">
          <p className="font-medium">
            Accuracy: {(trainResult.accuracy * 100).toFixed(0)}%
            {!trainResult.held_out_evaluation &&
              ' (train set — not enough data yet to hold out a test split)'}
          </p>
          <ConfusionMatrix matrix={trainResult.confusion_matrix} />
        </div>
      )}

      {status.model_trained && !trainResult && (
        <p className="text-xs text-gray-500">
          Model trained at{' '}
          {status.trained_at ? new Date(status.trained_at * 1000).toLocaleTimeString() : '—'}
          {status.accuracy !== null && ` · ${(status.accuracy * 100).toFixed(0)}% accuracy`}
        </p>
      )}
    </div>
  );
};

const ConfusionMatrix: React.FC<{ matrix: TrainResult['confusion_matrix'] }> = ({ matrix }) => (
  <div className="overflow-x-auto mt-2">
    <table className="text-xs border-collapse">
      <thead>
        <tr>
          <th className="p-1" />
          {matrix.labels.map((l) => (
            <th key={l} className="p-1 font-normal text-gray-500">
              {l}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.matrix.map((row, i) => (
          <tr key={matrix.labels[i]}>
            <td className="p-1 font-medium text-gray-500">{matrix.labels[i]}</td>
            {row.map((val, j) => (
              <td
                key={j}
                className={`p-1 text-center border ${i === j ? 'bg-green-50 font-medium' : ''}`}
              >
                {val}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

export default GestureTrainer;

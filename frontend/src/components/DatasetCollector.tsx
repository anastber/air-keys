'use client';

// Dev tool for collecting the author's own labeled dataset (own_v2 — see
// ROADMAP.md, Phase 0.2): the only data captured in the real deployment
// condition (live webcam through the in-browser pipeline), used as the
// deployment-condition test set alongside the multi-subject HaGRID data.
// Not part of the production UI (see app/collect/page.tsx) — deliberately
// separate from GestureTrainer.tsx, whose recordings go straight into
// localStorage for the client-side kNN. This one accumulates raw samples in
// memory across many labels/recording rounds and exports them as a single
// JSON file, using the exact same MediaPipe pipeline (WebcamLandmarks) as
// production inference.
//
// Every sample is tagged with a recording session (plus camera and
// lighting), so evaluation can hold out whole sessions — frames from one
// session share a background, light and camera position, and splitting them
// across train and test would leak in the same way splitting a burst does.

import React, { useCallback, useEffect, useState } from 'react';
import WebcamLandmarks from '@/components/WebcamLandmarks';
import { gestureEmoji } from '@/lib/gestureIcons';
import type { CameraInfo, HandData, Landmark, LandmarkData } from '@/lib/types';

const SAMPLES_PER_RECORDING = 20;
const TARGET_SAMPLES_PER_CLASS = 250;

const SCHEMA_VERSION = 2;

// Same labels as the HaGRID extraction (ml/datasets/hagrid_labels.json), so
// a model trained on HaGRID can be evaluated on every class recorded here.
const TARGET_CLASSES: { label: string; description: string; hardNegatives?: string[] }[] = [
  { label: 'ok_sign', description: 'Thumb + index touching in a circle, other 3 fingers up' },
  { label: 'rock_on', description: 'Index + pinky extended, thumb/middle/ring folded' },
  { label: 'call_me', description: 'Thumb + pinky extended, other 3 fingers folded' },
  { label: 'fist', description: 'All fingers folded' },
  { label: 'open_palm', description: 'All fingers extended, palm facing the camera' },
  { label: 'peace', description: 'Index + middle extended in a V' },
  { label: 'point', description: 'Index extended upward, others folded' },
  { label: 'thumbs_up', description: 'Thumb up, other fingers folded' },
  {
    label: 'no_gesture',
    description:
      'Relaxed, curled or mid-transition hand — none of the classes above — plus ' +
      'deliberately holding a pinch, which the app handles with geometry, not this model:',
    hardNegatives: ['pinch'],
  },
];

const LIGHTING_OPTIONS = ['daylight', 'artificial', 'dim', 'backlit'] as const;
type Lighting = (typeof LIGHTING_OPTIONS)[number];

// Session fields are optional so datasets exported before schema v2 (no
// sessions) can still be loaded and extended.
interface DatasetRecord {
  schema_version?: number;
  source?: 'own';
  label: string;
  landmarks: Landmark[];
  handedness: string;
  timestamp: number;
  session_id?: string;
  camera?: string;
  resolution?: { width: number; height: number };
  lighting?: Lighting;
}

function newSessionId(): string {
  const date = new Date().toISOString().slice(0, 10);
  return `${date}-${Math.random().toString(36).slice(2, 6)}`;
}

const DatasetCollector: React.FC = () => {
  const [labelInput, setLabelInput] = useState('');
  const [recordingLabel, setRecordingLabel] = useState<string | null>(null);
  const [recordedCount, setRecordedCount] = useState(0);
  const [samples, setSamples] = useState<DatasetRecord[]>([]);
  const [currentHand, setCurrentHand] = useState<HandData | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [lighting, setLighting] = useState<Lighting | null>(null);
  const [camera, setCamera] = useState<CameraInfo | null>(null);

  // Generated after mount, not during render: a random id computed on the
  // server would differ from the client's and break hydration.
  useEffect(() => setSessionId(newSessionId()), []);

  const startNewSession = () => {
    setSessionId(newSessionId());
    setLighting(null);
  };

  const handleLandmarks = useCallback((data: LandmarkData) => {
    setCurrentHand(data.hands[0] ?? null);
  }, []);

  // While recording, capture one sample per incoming frame until the target
  // count is hit — same shape as GestureTrainer.tsx's capture effect, just
  // appending to in-memory state instead of writing to localStorage.
  useEffect(() => {
    if (!recordingLabel || !currentHand || currentHand.landmarks.length !== 21) return;
    if (!sessionId || !lighting || !camera) return;

    if (recordedCount >= SAMPLES_PER_RECORDING) {
      setRecordingLabel(null);
      return;
    }

    setSamples((prev) => [
      ...prev,
      {
        schema_version: SCHEMA_VERSION,
        source: 'own',
        label: recordingLabel,
        landmarks: currentHand.landmarks,
        handedness: currentHand.handedness,
        timestamp: Date.now() / 1000,
        session_id: sessionId,
        camera: camera.label,
        resolution: { width: camera.width, height: camera.height },
        lighting,
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
    a.download = 'own_v2.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Resume a previous export (e.g. ml/data/own_v2.json) so adding more
  // samples — new classes, or more of an existing one — doesn't mean
  // re-recording everything from scratch.
  const loadDataset = async (file: File) => {
    const loaded = JSON.parse(await file.text()) as DatasetRecord[];
    setSamples((prev) => [...prev, ...loaded]);
  };

  const counts: Record<string, number> = {};
  for (const s of samples) counts[s.label] = (counts[s.label] ?? 0) + 1;

  const sessionCounts: Record<string, number> = {};
  for (const s of samples) {
    const id = s.session_id ?? 'no session (pre-v2)';
    sessionCounts[id] = (sessionCounts[id] ?? 0) + 1;
  }

  const progress = recordingLabel ? recordedCount / SAMPLES_PER_RECORDING : 0;
  const canRecord = Boolean(currentHand && labelInput.trim() && sessionId && lighting && camera);

  return (
    <main className="min-h-screen bg-ak-bg text-ak-text p-6 flex flex-col gap-6 max-w-2xl mx-auto">
      <div>
        <h1 className="text-lg font-semibold">Dataset collector (dev tool)</h1>
        <p className="text-sm text-ak-muted mt-1">
          Record the 9 classes below over several sessions, then save the export to{' '}
          <code>ml/data/own_v2.json</code>.
        </p>
        <label className="inline-block mt-2 text-xs text-ak-accent hover:underline cursor-pointer">
          Load an existing dataset to add more samples on top
          <input
            type="file"
            accept="application/json"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void loadDataset(file);
              e.target.value = '';
            }}
          />
        </label>
      </div>

      {/* Recording session — every sample is tagged with it. */}
      <div className="flex flex-col gap-2 rounded-lg border border-ak-border bg-ak-panel/40 px-4 py-3 text-sm">
        <div className="flex items-center justify-between gap-2">
          <span>
            Session <code className="text-ak-accent">{sessionId ?? '…'}</code>
          </span>
          <button
            onClick={startNewSession}
            disabled={recordingLabel !== null}
            className="text-xs text-ak-accent hover:underline disabled:opacity-50"
          >
            New session
          </button>
        </div>
        <label className="flex items-center gap-2">
          <span className="text-ak-muted">Lighting</span>
          <select
            value={lighting ?? ''}
            onChange={(e) => setLighting((e.target.value || null) as Lighting | null)}
            disabled={recordingLabel !== null}
            className="bg-transparent border border-ak-border rounded-md px-2 py-1 text-sm disabled:opacity-50"
          >
            <option value="">choose…</option>
            {LIGHTING_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <span className="text-ak-muted">
          Camera:{' '}
          {camera ? `${camera.label} · ${camera.width}×${camera.height}` : 'waiting for webcam…'}
        </span>
        {Object.keys(sessionCounts).length > 0 && (
          <span className="text-xs text-ak-subtle">
            {Object.keys(sessionCounts).length} session(s) in dataset:{' '}
            {Object.entries(sessionCounts)
              .map(([id, count]) => `${id} (${count})`)
              .join(', ')}
          </span>
        )}
      </div>

      {/* Target classes — click one to fill the label input below. */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {TARGET_CLASSES.map(({ label, description, hardNegatives }) => {
          const count = samples.filter((s) => s.label === label).length;
          const done = count >= TARGET_SAMPLES_PER_CLASS;
          return (
            <button
              key={label}
              onClick={() => setLabelInput(label)}
              disabled={recordingLabel !== null}
              className={`text-left flex flex-col gap-1 rounded-lg border px-3 py-2.5 transition-colors disabled:opacity-50 ${
                labelInput === label
                  ? 'border-ak-accent/60 bg-ak-accent/10'
                  : 'border-ak-border bg-ak-panel/40 hover:bg-ak-panel'
              }`}
            >
              <span className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-sm">
                  <span className="text-lg leading-none">{gestureEmoji(label)}</span>
                  {label}
                </span>
                <span className={`text-xs ${done ? 'text-ak-emerald' : 'text-ak-subtle'}`}>
                  {count}/{TARGET_SAMPLES_PER_CLASS}
                </span>
              </span>
              <span className="text-xs text-ak-muted">{description}</span>
              {hardNegatives && (
                <span className="flex items-center gap-2 text-xl leading-none pt-0.5">
                  {hardNegatives.map((g) => (
                    <span key={g} title={g}>
                      {gestureEmoji(g)}
                    </span>
                  ))}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* How-to, kept short — the counters above are the real feedback loop. */}
      <ol className="text-xs text-ak-muted list-decimal list-inside flex flex-col gap-1 bg-ak-panel/40 border border-ak-border rounded-lg px-4 py-3">
        <li>
          One session = one sitting (same day, place and light). Aim for 5+ sessions on
          different days and lighting; start a <strong>New session</strong> each time and set its
          lighting before recording.
        </li>
        <li>Click a class above to fill its label, then hit Record and hold the pose.</li>
        <li>Each Record captures ~1s (20 frames) — repeat a few times per class per session.</li>
        <li>
          Between recordings, change distance, rotation, hand position in frame, and
          which hand — a single long hold gives near-identical frames.
        </li>
        <li>
          <code>no_gesture</code> needs the most variety: relaxed/curled hand, mid-transition
          poses, different distances — this is what stops false triggers at rest — plus a{' '}
          {gestureEmoji('pinch')} pinch, which must not be confused with any class.
        </li>
        <li>Download the dataset once every class shows ~250+ samples.</li>
      </ol>

      <div className="rounded-lg overflow-hidden border border-ak-border">
        <WebcamLandmarks onLandmarks={handleLandmarks} onCameraInfo={setCamera} />
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={labelInput}
          onChange={(e) => setLabelInput(e.target.value)}
          placeholder="e.g. ok_sign"
          disabled={recordingLabel !== null}
          className="flex-1 bg-transparent border border-ak-border rounded-md px-3 py-1.5 text-sm placeholder:text-ak-subtle focus:outline-none focus:ring-1 focus:ring-ak-accent/60 disabled:opacity-50"
        />
        <button
          onClick={startRecording}
          disabled={!canRecord || recordingLabel !== null}
          className="px-4 py-1.5 rounded-md text-sm font-medium bg-ak-line text-ak-bg disabled:opacity-40"
        >
          Record
        </button>
      </div>

      {recordingLabel && (
        <p className="text-sm text-ak-accent">
          &ldquo;{recordingLabel}&rdquo; — {recordedCount}/{SAMPLES_PER_RECORDING} ({Math.round(progress * 100)}%)
        </p>
      )}
      {!lighting && <p className="text-xs text-ak-amber">Set this session&apos;s lighting to record.</p>}
      {!currentHand && <p className="text-xs text-ak-amber">Show a hand to the camera to record.</p>}

      {Object.keys(counts).length > 0 && (
        <div className="flex flex-col gap-1.5 pt-2 border-t border-ak-border">
          {Object.entries(counts).map(([label, count]) => (
            <div key={label} className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-1.5">
                <span>{gestureEmoji(label)}</span>
                <span>
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
            className="mt-2 self-start px-4 py-1.5 rounded-md text-sm font-medium bg-ak-panel hover:bg-ak-panel-hover border border-ak-border"
          >
            Download dataset ({samples.length} samples)
          </button>
        </div>
      )}
    </main>
  );
};

export default DatasetCollector;

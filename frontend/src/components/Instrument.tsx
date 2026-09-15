'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as Tone from 'tone';
import WebcamLandmarks from '@/components/WebcamLandmarks';
import GestureTrainer from '@/components/GestureTrainer';
import RuleEditor from '@/components/RuleEditor';
import { isPinching, noteForHeight as legacyNoteForHeight } from '@/lib/notes';
import { DEFAULT_RULES, resolveAction, type GestureRule } from '@/lib/rules';
import { fetchGestureStatus, type GestureStatus } from '@/lib/gestureApi';
import type { LandmarkData } from '@/lib/types';

// Ties the three layers together: WebcamLandmarks (perception), the gesture
// classifier's output riding along on the same stream, and rules (control)
// that turn a recognized gesture into a Tone.js action.
//
// Until a classifier is trained, hands.gesture is absent on every message
// (see backend.api.main), so playback falls back to the original raw-pinch
// behavior — the app is always playable, gestures just get richer once
// you've trained some.
const Instrument: React.FC = () => {
  const [rules, setRules] = useState<Record<string, GestureRule>>(DEFAULT_RULES);
  const [status, setStatus] = useState<GestureStatus | null>(null);
  const [currentHand, setCurrentHand] = useState<LandmarkData['hands'][number] | null>(null);
  const [isAudioEnabled, setIsAudioEnabled] = useState(false);
  const [lastAction, setLastAction] = useState<string | null>(null);

  const synthRef = useRef<Tone.PolySynth | null>(null);
  const gestureStateRef = useRef<Record<string, string | null>>({});
  const pinchStateRef = useRef<Record<string, boolean>>({});
  const sustainRef = useRef(false);

  useEffect(() => {
    fetchGestureStatus()
      .then(setStatus)
      .catch(() => {
        // No trained model / backend not up yet — the trainer/editor just
        // stay hidden until status is available.
      });
  }, []);

  const enableAudio = useCallback(async () => {
    await Tone.start();
    if (!synthRef.current) {
      synthRef.current = new Tone.PolySynth(Tone.Synth).toDestination();
    }
    setIsAudioEnabled(true);
  }, []);

  const handleLandmarks = useCallback(
    (data: LandmarkData) => {
      setCurrentHand(data.hands[0] ?? null);
      if (!isAudioEnabled || !synthRef.current) return;

      const duration = sustainRef.current ? '1n' : '8n';

      data.hands.forEach((hand) => {
        if (hand.landmarks.length < 21) return;
        const wrist = hand.landmarks[0];

        if (hand.gesture) {
          // Trained-classifier path: act once per gesture change, not every
          // frame the gesture is held (otherwise a pose re-triggers ~15x/sec).
          const prevGesture = gestureStateRef.current[hand.handedness] ?? null;
          if (hand.gesture !== prevGesture) {
            const rule = rules[hand.gesture];
            if (rule) {
              const action = resolveAction(rule, wrist.y);
              if (action.type === 'sustain_toggle') {
                sustainRef.current = !sustainRef.current;
                setLastAction(`${hand.gesture} -> sustain ${sustainRef.current ? 'on' : 'off'}`);
              } else if (action.type === 'arpeggio') {
                action.notes.forEach((note, i) => {
                  setTimeout(() => synthRef.current?.triggerAttackRelease(note, '16n'), i * 90);
                });
                setLastAction(`${hand.gesture} -> arpeggio: ${action.notes.join(' ')}`);
              } else {
                synthRef.current?.triggerAttackRelease(action.notes, duration);
                setLastAction(`${hand.gesture} -> ${action.type}: ${action.notes.join(' ')}`);
              }
            }
          }
          gestureStateRef.current[hand.handedness] = hand.gesture;
        } else {
          // Legacy fallback while no classifier is trained yet.
          const thumbTip = hand.landmarks[4];
          const indexTip = hand.landmarks[8];
          const pinched = isPinching(thumbTip, indexTip);
          const wasPinched = pinchStateRef.current[hand.handedness] ?? false;
          if (pinched && !wasPinched) {
            const note = legacyNoteForHeight(wrist.y);
            synthRef.current?.triggerAttackRelease(note, '8n');
            setLastAction(`pinch -> note: ${note}`);
          }
          pinchStateRef.current[hand.handedness] = pinched;
        }
      });
    },
    [isAudioEnabled, rules]
  );

  const updateRule = useCallback((label: string, patch: Partial<GestureRule>) => {
    setRules((prev) => ({ ...prev, [label]: { ...prev[label], ...patch } }));
  }, []);

  return (
    <div className="flex flex-col items-center gap-6 p-4">
      <h2 className="text-2xl font-bold text-center">AirKeys</h2>

      {!isAudioEnabled && (
        <button
          onClick={enableAudio}
          className="px-4 py-2 rounded font-medium bg-blue-500 hover:bg-blue-600 text-white"
        >
          Enable Sound
        </button>
      )}

      <WebcamLandmarks onLandmarks={handleLandmarks} />

      {lastAction && <div className="text-sm font-mono text-gray-700">{lastAction}</div>}

      <div className="text-sm text-gray-600 max-w-md text-center">
        {status?.model_trained ? (
          <p>The gesture rules below drive playback. Retrain any time as you add more samples.</p>
        ) : (
          <p>
            No gesture model trained yet — pinch your thumb and index finger to play a note
            (hand height sets pitch) while you build a training set below.
          </p>
        )}
      </div>

      <div className="flex flex-wrap justify-center gap-4">
        <GestureTrainer currentHand={currentHand} status={status} onStatusChange={setStatus} />
        <RuleEditor labels={status?.labels ?? []} rules={rules} onChange={updateRule} />
      </div>
    </div>
  );
};

export default Instrument;

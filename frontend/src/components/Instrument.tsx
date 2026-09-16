'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as Tone from 'tone';
import WebcamLandmarks from '@/components/WebcamLandmarks';
import GestureTrainer from '@/components/GestureTrainer';
import RuleEditor from '@/components/RuleEditor';
import ComboGuide from '@/components/ComboGuide';
import { isPinching } from '@/lib/notes';
import { predictCustomGesture, customGestureLabels } from '@/lib/customGestures';
import {
  BASE_GESTURE_LABELS,
  COMBOS,
  DEFAULT_CUSTOM_RULE,
  DEFAULT_RULES,
  matchCombo,
  resolveAction,
  type Combo,
  type GestureRule,
} from '@/lib/rules';
import type { LandmarkData } from '@/lib/types';

// How long a partial combo stays "live" before it's forgotten — long enough
// to perform 3 deliberate gestures, short enough that idle play doesn't
// coincidentally complete one.
const COMBO_WINDOW_MS = 6000;

// Ties the layers together, per hand, per frame, in one priority chain:
//   1. a gesture the visitor taught themselves (client-side kNN, private to
//      their browser) — takes priority since it's the one they chose to add
//   2. pinch — pure thumb-index distance, no model at all, always available
//   3. MediaPipe's pretrained canned pose — works for anyone, zero setup
// Whichever resolves fires a rule (lib/rules.ts) once per gesture *change*,
// not every frame it's held.
const Instrument: React.FC = () => {
  const [rules, setRules] = useState<Record<string, GestureRule>>(DEFAULT_RULES);
  const [customLabels, setCustomLabels] = useState<string[]>([]);
  const [currentHand, setCurrentHand] = useState<LandmarkData['hands'][number] | null>(null);
  const [isAudioEnabled, setIsAudioEnabled] = useState(false);
  const [lastAction, setLastAction] = useState<string | null>(null);

  const synthRef = useRef<Tone.PolySynth | null>(null);
  const gestureStateRef = useRef<Record<string, string | null>>({});
  const sustainRef = useRef(false);
  const recentGesturesRef = useRef<{ label: string; time: number }[]>([]);

  const playCombo = useCallback((combo: Combo) => {
    let delay = 200; // let the triggering gesture's own note ring briefly first
    combo.melody.forEach((step) => {
      setTimeout(() => synthRef.current?.triggerAttackRelease(step.notes, step.duration), delay);
      delay += combo.stepDelayMs;
    });
  }, []);

  useEffect(() => {
    setCustomLabels(customGestureLabels());
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

        // Priority chain: taught gesture > pinch > canned pose > nothing.
        const custom = predictCustomGesture(hand.landmarks, hand.handedness);
        const thumbTip = hand.landmarks[4];
        const indexTip = hand.landmarks[8];
        const label = custom
          ? custom.label
          : isPinching(thumbTip, indexTip)
            ? 'pinch'
            : (hand.gesture ?? null);

        const prevLabel = gestureStateRef.current[hand.handedness] ?? null;
        if (label !== prevLabel && label) {
          const rule = rules[label];
          if (rule) {
            const action = resolveAction(rule, wrist.y);
            if (action.type === 'sustain_toggle') {
              sustainRef.current = !sustainRef.current;
              setLastAction(`${label} -> sustain ${sustainRef.current ? 'on' : 'off'}`);
            } else if (action.type === 'arpeggio') {
              action.notes.forEach((note, i) => {
                setTimeout(() => synthRef.current?.triggerAttackRelease(note, '16n'), i * 90);
              });
              setLastAction(`${label} -> arpeggio: ${action.notes.join(' ')}`);
            } else {
              synthRef.current?.triggerAttackRelease(action.notes, duration);
              setLastAction(`${label} -> ${action.type}: ${action.notes.join(' ')}`);
            }
          }

          // Combo tracking: a single timeline shared across both hands — do
          // 3 specific gestures in order, from either hand, within the
          // window, and a bonus tune plays on top of the action above.
          const now = Date.now();
          const recent = [...recentGesturesRef.current, { label, time: now }].filter(
            (e) => now - e.time < COMBO_WINDOW_MS
          );
          recentGesturesRef.current = recent.slice(-6);
          const combo = matchCombo(recent.map((e) => e.label));
          if (combo) {
            recentGesturesRef.current = [];
            playCombo(combo);
            setLastAction(`🎶 combo: ${combo.name}!`);
          }
        }
        gestureStateRef.current[hand.handedness] = label;
      });
    },
    [isAudioEnabled, rules, playCombo]
  );

  const updateRule = useCallback((label: string, patch: Partial<GestureRule>) => {
    setRules((prev) => ({ ...prev, [label]: { ...prev[label], ...patch } }));
  }, []);

  // Give a newly taught gesture a default rule so it shows up (and does
  // something) in the editor immediately, without waiting for manual setup.
  const handleGestureRecorded = useCallback((label: string) => {
    setCustomLabels(customGestureLabels());
    setRules((prev) => (prev[label] ? prev : { ...prev, [label]: DEFAULT_CUSTOM_RULE }));
  }, []);

  const allLabels = [...BASE_GESTURE_LABELS, ...customLabels];

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

      <p className="text-sm text-gray-600 max-w-md text-center">
        Pinch, fist, open palm, point, peace, or thumbs up all work immediately — no
        setup. Teach it a gesture of your own below to add to the set.
      </p>

      <div className="flex flex-wrap justify-center gap-4">
        <ComboGuide combos={COMBOS} />
        <GestureTrainer currentHand={currentHand} onGestureRecorded={handleGestureRecorded} />
        <RuleEditor labels={allLabels} rules={rules} onChange={updateRule} />
      </div>
    </div>
  );
};

export default Instrument;

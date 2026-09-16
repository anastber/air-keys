'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as Tone from 'tone';
import WebcamLandmarks from '@/components/WebcamLandmarks';
import GestureTrainer from '@/components/GestureTrainer';
import RuleEditor from '@/components/RuleEditor';
import ComboGuide from '@/components/ComboGuide';
import AudioVisualizer from '@/components/AudioVisualizer';
import { isPinching } from '@/lib/notes';
import { gestureEmoji } from '@/lib/gestureIcons';
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

const ACTION_ICON: Record<string, string> = {
  note: '🎵',
  chord: '🎹',
  bass: '🎸',
  arpeggio: '🎼',
};

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
  const [analyser, setAnalyser] = useState<Tone.Analyser | null>(null);
  const [lastAction, setLastAction] = useState<{ text: string; icon: string } | null>(null);
  const [actionSeq, setActionSeq] = useState(0);

  const synthRef = useRef<Tone.PolySynth | null>(null);
  const gestureStateRef = useRef<Record<string, string | null>>({});
  const sustainRef = useRef(false);
  const recentGesturesRef = useRef<{ label: string; time: number }[]>([]);

  // Single source of truth for "something just happened": drives both the
  // toast text and the glow pulse around the video (keyed by actionSeq).
  const announce = useCallback((text: string, icon: string) => {
    setLastAction({ text, icon });
    setActionSeq((s) => s + 1);
  }, []);

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
      const synth = new Tone.PolySynth(Tone.Synth).toDestination();
      const fft = new Tone.Analyser('fft', 64);
      synth.connect(fft);
      synthRef.current = synth;
      setAnalyser(fft);
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
            const emoji = gestureEmoji(label);
            if (action.type === 'sustain_toggle') {
              sustainRef.current = !sustainRef.current;
              announce(
                `${emoji} ${label} → sustain ${sustainRef.current ? 'on' : 'off'}`,
                sustainRef.current ? '🔊' : '🔈'
              );
            } else if (action.type === 'arpeggio') {
              action.notes.forEach((note, i) => {
                setTimeout(() => synthRef.current?.triggerAttackRelease(note, '16n'), i * 90);
              });
              announce(`${emoji} ${label} → arpeggio: ${action.notes.join(' ')}`, '🎼');
            } else {
              synthRef.current?.triggerAttackRelease(action.notes, duration);
              announce(
                `${emoji} ${label} → ${action.type}: ${action.notes.join(' ')}`,
                ACTION_ICON[action.type] ?? '🎵'
              );
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
            announce(
              `${combo.name}: ${combo.sequence.map(gestureEmoji).join(' ')}`,
              '🎶'
            );
          }
        }
        gestureStateRef.current[hand.handedness] = label;
      });
    },
    [isAudioEnabled, rules, playCombo, announce]
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
    <div className="flex flex-col items-center gap-8 px-4 py-10 max-w-6xl mx-auto">
      <header className="flex flex-col items-center gap-2 text-center">
        <span className="ak-glass rounded-full px-3 py-1 text-xs text-ak-muted tracking-wide uppercase">
          Client-side AI · zero setup
        </span>
        <h1 className="text-5xl sm:text-6xl font-bold ak-gradient-text tracking-tight py-1">
          AirKeys
        </h1>
        <p className="text-ak-muted max-w-md">
          Wave, pinch, or fist-bump the air — your webcam turns it into music.
        </p>
      </header>

      {!isAudioEnabled && (
        <button
          onClick={enableAudio}
          className="relative px-6 py-3 rounded-full font-medium text-white bg-gradient-to-r from-ak-violet via-fuchsia-500 to-ak-cyan shadow-lg shadow-ak-violet/30 hover:scale-105 active:scale-100 transition-transform animate-glow-pulse"
        >
          🔈 Enable Sound to Begin
        </button>
      )}

      <div className="relative">
        {/* Glow ring that flashes around the video on every trigger, keyed
            by actionSeq so each new trigger restarts the animation. Gated on
            actionSeq > 0 so it doesn't also fire once on initial mount. */}
        {actionSeq > 0 && (
          <div
            key={actionSeq}
            className="pointer-events-none absolute -inset-3 rounded-[2rem] animate-glow-pulse"
          />
        )}
        <WebcamLandmarks onLandmarks={handleLandmarks} />
      </div>

      <div className="w-full max-w-[640px] ak-glass rounded-2xl px-3 py-2">
        <AudioVisualizer analyser={analyser} />
      </div>

      {lastAction && (
        <div
          key={actionSeq}
          className="ak-glass flex items-center gap-2 rounded-full px-4 py-2 text-sm font-mono text-ak-text animate-fade-in-up"
        >
          <span className="text-base">{lastAction.icon}</span>
          {lastAction.text}
        </div>
      )}

      <p className="text-sm text-ak-muted max-w-md text-center">
        Pinch, fist, open palm, point, peace, or thumbs up all work immediately — no
        setup. Teach it a gesture of your own below to add to the set.
      </p>

      <div className="flex flex-wrap justify-center gap-4 w-full">
        <ComboGuide combos={COMBOS} />
        <GestureTrainer currentHand={currentHand} onGestureRecorded={handleGestureRecorded} />
        <RuleEditor labels={allLabels} rules={rules} onChange={updateRule} />
      </div>
    </div>
  );
};

export default Instrument;

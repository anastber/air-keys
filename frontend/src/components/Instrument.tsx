'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as Tone from 'tone';
import WebcamLandmarks, { type StageStatus } from '@/components/WebcamLandmarks';
import ControlDeck from '@/components/ControlDeck';
import AudioVisualizer from '@/components/AudioVisualizer';
import { isPinching } from '@/lib/notes';
import { gestureEmoji } from '@/lib/gestureIcons';
import { predictCustomGesture, customGestureLabels } from '@/lib/customGestures';
import { loadTrainedGestureModel, predictTrainedGesture } from '@/lib/trainedGestures';
import {
  BASE_GESTURE_LABELS,
  COMBOS,
  DEFAULT_CUSTOM_RULE,
  DEFAULT_RULES,
  TRAINED_GESTURE_LABELS,
  matchCombo,
  resolveAction,
  type Combo,
  type GestureRule,
} from '@/lib/rules';
import { SONGS } from '@/lib/songs';
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

// Layout, top to bottom: a compact header, then a wide "stage" (the camera
// feed — the actual point of the app) paired with a tabbed "control deck"
// for the three secondary panels (combos / teach / rules), side by side on
// desktop and stacked stage-first on mobile. Rationale for putting the
// three panels behind tabs rather than as three stacked cards: they serve
// different moments (reference vs. input vs. configuration) and don't need
// to all be visible while someone is just trying to play.
//
// Gesture-to-audio priority chain, per hand, per frame:
//   1. a gesture the visitor taught themselves (client-side kNN, private to
//      their browser) — takes priority since it's the one they chose to add
//   2. a self-trained MLP's pose (ok_sign/rock_on/call_me — see
//      lib/trainedGestures.ts), checked before pinch specifically so a
//      confident ok_sign (thumb+index touching) wins over the more generic
//      pinch geometry check below, which would otherwise also fire for it
//   3. pinch — pure thumb-index distance, no model at all, always available
//   4. MediaPipe's pretrained canned pose — works for anyone, zero setup
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
  const [stageStatus, setStageStatus] = useState<StageStatus>({
    isWebcamActive: false,
    isModelReady: false,
    error: null,
    handCount: 0,
  });

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

  // Fire-and-forget: predictTrainedGesture no-ops (returns null) until this
  // resolves, same tolerance WebcamLandmarks has for the MediaPipe model
  // still loading.
  useEffect(() => {
    loadTrainedGestureModel().catch((err) => {
      console.error('Failed to load trained gesture model:', err);
    });
  }, []);

  const enableAudio = useCallback(async () => {
    await Tone.start();
    if (!synthRef.current) {
      // A quick attack keeps single melody notes crisp (important now that
      // most gestures play one fixed note rather than a sustained chord),
      // and a touch of reverb gives them some room instead of sounding dry.
      const synth = new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'triangle' },
        envelope: { attack: 0.005, decay: 0.15, sustain: 0.25, release: 0.6 },
      });
      const reverb = new Tone.Reverb({ decay: 1.8, wet: 0.22 });
      await reverb.generate();
      synth.connect(reverb);
      reverb.toDestination();

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

        // Priority chain: taught gesture > trained model > pinch > canned pose > nothing.
        const custom = predictCustomGesture(hand.landmarks, hand.handedness);
        const trained = custom ? null : predictTrainedGesture(hand.landmarks, hand.handedness);
        const thumbTip = hand.landmarks[4];
        const indexTip = hand.landmarks[8];
        const label = custom
          ? custom.label
          : trained
            ? trained.label
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
            announce(`${combo.name}: ${combo.sequence.map(gestureEmoji).join(' ')}`, '🎶');
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

  const allLabels = [...BASE_GESTURE_LABELS, ...TRAINED_GESTURE_LABELS, ...customLabels];

  return (
    <div className="flex flex-col items-center gap-6 px-4 py-8 max-w-6xl mx-auto">
      <header className="flex flex-col items-center gap-1.5 text-center">
        <span className="ak-glass rounded-full px-3 py-1 text-[11px] text-ak-muted tracking-wide uppercase">
          Client-side AI · zero setup
        </span>
        <h1 className="text-4xl sm:text-5xl font-bold ak-gradient-text tracking-tight py-1">
          AirKeys
        </h1>
        <p className="text-ak-muted text-sm max-w-md">
          Wave, pinch, or fist-bump the air — your webcam turns it into music.
        </p>
      </header>

      <div className="w-full grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-6 items-start">
        {/* Stage: the actual instrument */}
        <div className="flex flex-col gap-3">
          <div className="relative">
            {/* Glow ring that flashes around the stage on every trigger,
                keyed by actionSeq so each new trigger restarts the
                animation. Gated on actionSeq > 0 so it doesn't also fire
                once on initial mount. */}
            {actionSeq > 0 && (
              <div
                key={actionSeq}
                className="pointer-events-none absolute -inset-3 rounded-[2.5rem] animate-glow-pulse z-10"
              />
            )}

            <div className="ak-glass rounded-3xl overflow-hidden shadow-[0_0_60px_-15px_rgba(168,85,247,0.35)]">
              <div className="relative">
                <WebcamLandmarks onLandmarks={handleLandmarks} onStatusChange={setStageStatus} />

                {/* Error takes precedence over the "tap to start" prompt.
                    Rendered before the status pills below so the dim/blur
                    layer sits *under* them, not on top washing them out. */}
                {stageStatus.error ? (
                  <div className="absolute inset-0 flex items-center justify-center bg-ak-bg/80 backdrop-blur-sm p-6 text-center">
                    <p className="text-ak-red text-sm max-w-xs">{stageStatus.error}</p>
                  </div>
                ) : (
                  !isAudioEnabled && (
                    <div className="absolute inset-0 flex items-center justify-center bg-ak-bg/60 backdrop-blur-sm">
                      <button
                        onClick={enableAudio}
                        className="px-6 py-3 rounded-full font-medium text-white bg-gradient-to-r from-ak-violet via-fuchsia-500 to-ak-cyan shadow-lg shadow-ak-violet/30 hover:scale-105 active:scale-100 transition-transform animate-glow-pulse"
                      >
                        🔈 Tap to Start Playing
                      </button>
                    </div>
                  )
                )}

                {/* Status pills, overlaid top-left, above the dim layer */}
                <div className="absolute top-3 left-3 flex flex-wrap gap-1.5 max-w-[80%]">
                  <StatusPill
                    ok={stageStatus.isWebcamActive}
                    okLabel="Webcam active"
                    badLabel="Webcam inactive"
                  />
                  <StatusPill
                    ok={stageStatus.isModelReady}
                    okLabel="Model ready"
                    badLabel="Loading model…"
                    pending={!stageStatus.isModelReady}
                  />
                  {stageStatus.isWebcamActive && (
                    <span className="ak-glass rounded-full px-3 py-1 text-xs text-ak-muted">
                      {stageStatus.handCount === 0
                        ? 'No hands in frame'
                        : `${stageStatus.handCount} hand${stageStatus.handCount > 1 ? 's' : ''}`}
                    </span>
                  )}
                </div>

                {/* Last action, overlaid bottom-center so it never shifts layout */}
                {lastAction && (
                  <div
                    key={actionSeq}
                    className="absolute bottom-3 left-1/2 -translate-x-1/2 ak-glass flex items-center gap-2 rounded-full px-4 py-1.5 text-sm font-mono text-ak-text animate-fade-in-up max-w-[90%]"
                  >
                    <span className="text-base shrink-0">{lastAction.icon}</span>
                    <span className="truncate">{lastAction.text}</span>
                  </div>
                )}
              </div>

              {/* Visualizer strip, flush against the video, same card */}
              <div className="border-t border-white/[0.06] px-3 py-2">
                <AudioVisualizer analyser={analyser} />
              </div>
            </div>
          </div>

          <p className="text-sm text-ak-muted text-center px-2">
            Fist, point, peace, open palm, pinch, OK sign, rock on, and call me each play
            one fixed note (do through do) — cycle through them to play a real melody. See
            the Songs tab for ready-made ones, or teach it a gesture of your own in the deck.
          </p>
        </div>

        {/* Control deck: songs / combos / teach / rules */}
        <ControlDeck
          songs={SONGS}
          combos={COMBOS}
          currentHand={currentHand}
          onGestureRecorded={handleGestureRecorded}
          labels={allLabels}
          rules={rules}
          onRuleChange={updateRule}
        />
      </div>

      <footer className="text-xs text-ak-subtle text-center pt-2">
        Built with MediaPipe, Tone.js, and Next.js ·{' '}
        <a
          href="https://github.com/anastber/air-keys"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-ak-muted"
        >
          View source
        </a>
      </footer>
    </div>
  );
};

const StatusPill: React.FC<{
  ok: boolean;
  okLabel: string;
  badLabel: string;
  pending?: boolean;
}> = ({ ok, okLabel, badLabel, pending }) => (
  <span className="ak-glass flex items-center gap-1.5 rounded-full px-3 py-1 text-xs">
    <span
      className={`w-1.5 h-1.5 rounded-full shrink-0 ${
        ok ? 'bg-ak-emerald' : pending ? 'bg-ak-amber animate-pulse' : 'bg-ak-red'
      }`}
    />
    <span className={ok ? 'text-ak-text' : pending ? 'text-ak-amber' : 'text-ak-red'}>
      {ok ? okLabel : badLabel}
    </span>
  </span>
);

export default Instrument;

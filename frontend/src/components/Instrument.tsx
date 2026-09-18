'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as Tone from 'tone';
import WebcamLandmarks, { type StageStatus } from '@/components/WebcamLandmarks';
import ControlDeck from '@/components/ControlDeck';
import AudioVisualizer from '@/components/AudioVisualizer';
import { isPinching } from '@/lib/notes';
import { predictCustomGesture, customGestureLabels } from '@/lib/customGestures';
import { loadTrainedGestureModel, predictTrainedGesture } from '@/lib/trainedGestures';
import {
  BASE_GESTURE_LABELS,
  DEFAULT_CUSTOM_RULE,
  DEFAULT_RULES,
  TRAINED_GESTURE_LABELS,
  resolveAction,
  type GestureRule,
} from '@/lib/rules';
import { SONGS } from '@/lib/songs';
import type { LandmarkData } from '@/lib/types';

// Layout, top to bottom: a compact header, then a wide "stage" (the camera
// feed — the actual point of the app) paired with a tabbed "control deck"
// for the secondary panels (songs / teach / rules), side by side on
// desktop and stacked stage-first on mobile. Rationale for putting the
// panels behind tabs rather than as stacked cards: they serve different
// moments (reference vs. input vs. configuration) and don't need to all
// be visible while someone is just trying to play.
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
  const [lastAction, setLastAction] = useState<string | null>(null);
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

  // Single source of truth for "something just happened": drives the toast
  // text, keyed by actionSeq so its fade-in restarts on every trigger.
  const announce = useCallback((text: string) => {
    setLastAction(text);
    setActionSeq((s) => s + 1);
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
            if (action.type === 'sustain_toggle') {
              sustainRef.current = !sustainRef.current;
              announce(`${label} → sustain ${sustainRef.current ? 'on' : 'off'}`);
            } else if (action.type === 'arpeggio') {
              action.notes.forEach((note, i) => {
                setTimeout(() => synthRef.current?.triggerAttackRelease(note, '16n'), i * 90);
              });
              announce(`${label} → arpeggio: ${action.notes.join(' ')}`);
            } else {
              synthRef.current?.triggerAttackRelease(action.notes, duration);
              announce(`${label} → ${action.type}: ${action.notes.join(' ')}`);
            }
          }
        }
        gestureStateRef.current[hand.handedness] = label;
      });
    },
    [isAudioEnabled, rules, announce]
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
      <header className="flex flex-col items-center gap-2 text-center">
        <h1 className="font-display italic font-medium text-5xl sm:text-6xl text-ak-line tracking-tight py-1">
          AirKeys
        </h1>
        <p className="text-ak-muted text-sm max-w-md">
          Your hands are the instrument. Your webcam watches your gestures and turns
          them into music, live in the browser.
        </p>
      </header>

      <div className="w-full grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-6 items-start">
        {/* Stage: the actual instrument */}
        <div className="flex flex-col gap-3">
          <div className="bg-ak-panel border border-ak-line rounded-lg overflow-hidden">
            <div className="relative">
              <WebcamLandmarks onLandmarks={handleLandmarks} onStatusChange={setStageStatus} />

              {/* Error takes precedence over the "tap to start" prompt.
                  Rendered before the status pills below so the dim/blur
                  layer sits *under* them, not on top washing them out. */}
              {stageStatus.error ? (
                <div className="absolute inset-0 flex items-center justify-center bg-ak-bg/85 backdrop-blur-sm p-6 text-center">
                  <p className="text-ak-red text-sm max-w-xs">{stageStatus.error}</p>
                </div>
              ) : (
                !isAudioEnabled && (
                  <div className="absolute inset-0 flex items-center justify-center bg-ak-bg/70 backdrop-blur-sm">
                    <button
                      onClick={enableAudio}
                      className="px-7 py-3.5 rounded-md border border-ak-line text-ak-line font-display italic text-lg hover:bg-ak-line hover:text-ak-bg transition-colors"
                    >
                      Tap to start playing
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
                  className="absolute bottom-3 left-1/2 -translate-x-1/2 ak-glass rounded-full px-4 py-1.5 text-sm text-ak-line animate-fade-in-up max-w-[90%]"
                >
                  <span className="truncate">{lastAction}</span>
                </div>
              )}
            </div>

            {/* Visualizer strip, flush against the video, same card */}
            <div className="border-t border-ak-border px-3 py-2">
              <AudioVisualizer analyser={analyser} />
            </div>
          </div>

          <p className="text-sm text-ak-muted text-center px-2">
            Fist, point, peace, open palm, pinch, OK sign, rock on, and call me each play
            one fixed note (do through do) — cycle through them to play a real melody. See
            the Songs tab for ready-made ones, or teach it a gesture of your own in the deck.
          </p>
          <p className="text-xs text-ak-subtle text-center px-2">
            Six poses come from Google&apos;s pretrained MediaPipe hand-tracking model; OK sign,
            rock on, and call me from a small neural net I trained myself on ~2,500 self-collected
            samples (99.8% held-out accuracy); and anything you teach it is matched live by
            k-nearest-neighbors — all running client-side, in your browser.
          </p>
        </div>

        {/* Control deck: songs / teach / rules */}
        <ControlDeck
          songs={SONGS}
          currentHand={currentHand}
          onGestureRecorded={handleGestureRecorded}
          labels={allLabels}
          rules={rules}
          onRuleChange={updateRule}
        />
      </div>

      <footer className="w-full border-t border-ak-border pt-4 flex items-center justify-center gap-3 text-xs text-ak-subtle">
        <span>
          Built by{' '}
          <a
            href="https://github.com/anastber"
            target="_blank"
            rel="noopener noreferrer"
            className="text-ak-text hover:text-ak-accent underline underline-offset-2"
          >
            Anas Tber
          </a>
        </span>
        <span aria-hidden="true">·</span>
        <a
          href="https://github.com/anastber/air-keys"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 hover:text-ak-accent transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
          </svg>
          Source
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

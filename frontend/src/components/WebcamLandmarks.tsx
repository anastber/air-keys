'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { GestureRecognizer } from '@mediapipe/tasks-vision';
import { loadGestureRecognizer, recognizeFrame } from '@/lib/gestureRecognition';
import { gestureEmoji } from '@/lib/gestureIcons';
import type { LandmarkData } from '@/lib/types';

// Right hand reads cyan, left reads pink — matches the app's accent palette
// instead of literal traffic-light red/blue.
const RIGHT_COLOR = '#22d3ee';
const LEFT_COLOR = '#f472b6';

interface WebcamLandmarksProps {
  // Fires on every recognized frame (including gesture classification from
  // the pretrained model). Audio/rule logic lives in whoever consumes this —
  // this component only does camera + perception, entirely client-side.
  onLandmarks?: (data: LandmarkData) => void;
}

const WebcamLandmarks: React.FC<WebcamLandmarksProps> = ({ onLandmarks }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recognizerRef = useRef<GestureRecognizer | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);

  const [isModelReady, setIsModelReady] = useState(false);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [landmarkData, setLandmarkData] = useState<LandmarkData | null>(null);

  // Load the gesture recognizer once (memoized in lib/gestureRecognition, so
  // this is cheap even across remounts) — separate from camera startup so a
  // slow model load doesn't block the webcam preview from showing.
  useEffect(() => {
    let cancelled = false;
    loadGestureRecognizer()
      .then((recognizer) => {
        if (!cancelled) {
          recognizerRef.current = recognizer;
          setIsModelReady(true);
        }
      })
      .catch((err) => {
        if (!cancelled) setError('Failed to load gesture model: ' + String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Start webcam
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          frameRate: 30
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsWebcamActive(true);
        setError(null);
      }
    } catch (error) {
      setError('Failed to access webcam: ' + String(error));
      setIsWebcamActive(false);
    }
  }, []);

  // Stop webcam
  const stopWebcam = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsWebcamActive(false);
    }
  }, []);

  // Draw landmarks on canvas
  const drawLandmarks = useCallback(() => {
    if (!canvasRef.current || !videoRef.current || !landmarkData) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const video = videoRef.current;

    // Set canvas size to match video
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections first so landmark points render on top of the lines.
    landmarkData.hands.forEach((hand) => {
      if (hand.landmarks.length !== 21) return;

      const connections = [
        // Thumb
        [0, 1], [1, 2], [2, 3], [3, 4],
        // Index finger
        [0, 5], [5, 6], [6, 7], [7, 8],
        // Middle finger
        [0, 9], [9, 10], [10, 11], [11, 12],
        // Ring finger
        [0, 13], [13, 14], [14, 15], [15, 16],
        // Pinky
        [0, 17], [17, 18], [18, 19], [19, 20],
        // Palm
        [5, 9], [9, 13], [13, 17]
      ];

      const color = hand.handedness === 'Right' ? RIGHT_COLOR : LEFT_COLOR;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;

      connections.forEach(([start, end]) => {
        const startLandmark = hand.landmarks[start];
        const endLandmark = hand.landmarks[end];

        ctx.beginPath();
        ctx.moveTo(startLandmark.x * canvas.width, startLandmark.y * canvas.height);
        ctx.lineTo(endLandmark.x * canvas.width, endLandmark.y * canvas.height);
        ctx.stroke();
      });
      ctx.shadowBlur = 0;
    });

    // Draw landmark points + hand label on top
    landmarkData.hands.forEach((hand) => {
      const color = hand.handedness === 'Right' ? RIGHT_COLOR : LEFT_COLOR;

      hand.landmarks.forEach((landmark, index) => {
        const x = landmark.x * canvas.width;
        const y = landmark.y * canvas.height;

        ctx.fillStyle = index === 0 ? color : '#ffffff';
        ctx.beginPath();
        ctx.arc(x, y, index === 0 ? 7 : 3, 0, 2 * Math.PI); // wrist is larger
        ctx.fill();
      });

      // Hand label: emoji + classified gesture once available, in a pill
      // that reads clearly over busy video backgrounds.
      if (hand.landmarks.length > 0) {
        const wrist = hand.landmarks[0];
        const labelX = wrist.x * canvas.width;
        const labelY = wrist.y * canvas.height - 24;

        const label = hand.gesture
          ? `${gestureEmoji(hand.gesture)} ${hand.gesture} ${Math.round((hand.confidence ?? 0) * 100)}%`
          : hand.handedness;

        ctx.font = '600 15px var(--font-geist-sans), system-ui, sans-serif';
        const textWidth = ctx.measureText(label).width;
        const paddingX = 10;
        const pillHeight = 26;

        ctx.fillStyle = 'rgba(9,7,15,0.75)';
        ctx.beginPath();
        ctx.roundRect(
          labelX - textWidth / 2 - paddingX,
          labelY - pillHeight / 2 - 10,
          textWidth + paddingX * 2,
          pillHeight,
          13
        );
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = '#f4f4f6';
        ctx.textAlign = 'center';
        ctx.fillText(label, labelX, labelY - 10 + pillHeight / 2 + 5);
        ctx.textAlign = 'left';
      }
    });
  }, [landmarkData]);

  // Effect to draw landmarks when data changes
  useEffect(() => {
    drawLandmarks();
  }, [drawLandmarks]);

  // Update canvas size when window resizes
  useEffect(() => {
    const handleResize = () => {
      drawLandmarks();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [drawLandmarks]);

  // Auto-start webcam on mount
  useEffect(() => {
    startWebcam();
  }, [startWebcam]);

  // The recognition loop: run entirely locally against the <video> element,
  // no server round-trip. Gated on video.currentTime actually advancing so
  // we don't reprocess the same frame twice when the render loop outpaces
  // the webcam's real frame rate.
  useEffect(() => {
    if (!isWebcamActive || !isModelReady) return;

    const tick = () => {
      const video = videoRef.current;
      const recognizer = recognizerRef.current;
      if (
        video &&
        recognizer &&
        video.readyState >= 2 &&
        video.currentTime !== lastVideoTimeRef.current
      ) {
        lastVideoTimeRef.current = video.currentTime;
        const hands = recognizeFrame(recognizer, video, performance.now());
        const data: LandmarkData = { timestamp: Date.now() / 1000, hands };
        setLandmarkData(data);
        onLandmarks?.(data);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [isWebcamActive, isModelReady, onLandmarks]);

  // Cleanup on unmount. Note: the recognizer itself is a module-level
  // singleton (see lib/gestureRecognition) and deliberately isn't closed
  // here — closing it would break a remount (e.g. React StrictMode's
  // mount/unmount/remount in dev) since the cached promise would resolve to
  // an already-closed instance.
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, [stopWebcam]);

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Status pills */}
      <div className="flex flex-wrap justify-center gap-2 text-xs">
        <StatusPill ok={isWebcamActive} okLabel="Webcam active" badLabel="Webcam inactive" />
        <StatusPill
          ok={isModelReady}
          okLabel="Gesture model ready"
          badLabel="Loading model…"
          pending={!isModelReady}
        />
        {landmarkData && (
          <span className="ak-glass rounded-full px-3 py-1 text-ak-muted">
            {landmarkData.hands.length === 0
              ? 'No hands in frame'
              : `${landmarkData.hands.length} hand${landmarkData.hands.length > 1 ? 's' : ''} tracked`}
          </span>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="bg-ak-red/10 border border-ak-red/40 text-ak-red px-4 py-3 rounded-xl max-w-md text-sm">
          {error}
        </div>
      )}

      {/* Video and Canvas container */}
      <div className="relative rounded-3xl p-1.5 ak-glass shadow-[0_0_60px_-15px_rgba(168,85,247,0.35)]">
        <div className="relative overflow-hidden rounded-2xl">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="block max-w-full h-auto scale-x-[-1]"
            style={{ maxWidth: '640px', maxHeight: '480px' }}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 pointer-events-none scale-x-[-1]"
          />
        </div>
      </div>
    </div>
  );
};

const StatusPill: React.FC<{
  ok: boolean;
  okLabel: string;
  badLabel: string;
  pending?: boolean;
}> = ({ ok, okLabel, badLabel, pending }) => (
  <span className="ak-glass flex items-center gap-1.5 rounded-full px-3 py-1">
    <span
      className={`w-1.5 h-1.5 rounded-full ${
        ok ? 'bg-ak-emerald' : pending ? 'bg-ak-amber animate-pulse' : 'bg-ak-red'
      }`}
    />
    <span className={ok ? 'text-ak-text' : pending ? 'text-ak-amber' : 'text-ak-red'}>
      {ok ? okLabel : badLabel}
    </span>
  </span>
);

export default WebcamLandmarks;

'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { GestureRecognizer } from '@mediapipe/tasks-vision';
import { loadGestureRecognizer, recognizeFrame } from '@/lib/gestureRecognition';
import type { LandmarkData } from '@/lib/types';

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

    // Draw landmarks for each detected hand
    landmarkData.hands.forEach((hand) => {
      const color = hand.handedness === 'Right' ? '#ff0000' : '#0000ff';


      hand.landmarks.forEach((landmark, index) => {
        // Convert normalized coordinates [0, 1] to canvas coordinates
        const x = landmark.x * canvas.width;
        const y = landmark.y * canvas.height;

        // Draw landmark point
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, index === 0 ? 8 : 4, 0, 2 * Math.PI); // Wrist is larger
        ctx.fill();

        // Draw landmark index for debugging
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.strokeText(index.toString(), x + 8, y - 8);
        ctx.fillText(index.toString(), x + 8, y - 8);
      });

      // Draw hand label (handedness, plus classified gesture once available)
      if (hand.landmarks.length > 0) {
        const wrist = hand.landmarks[0];
        const labelX = wrist.x * canvas.width;
        const labelY = wrist.y * canvas.height - 20;

        const label = hand.gesture
          ? `${hand.handedness} · ${hand.gesture} (${Math.round((hand.confidence ?? 0) * 100)}%)`
          : hand.handedness;

        ctx.fillStyle = color;
        ctx.font = '16px Arial';
        ctx.fillText(label, labelX, labelY);
      }
    });

    // Draw connections between landmarks (MediaPipe hand connections)
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

      const color = hand.handedness === 'Right' ? '#ff0000' : '#0000ff';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;

      connections.forEach(([start, end]) => {
        const startLandmark = hand.landmarks[start];
        const endLandmark = hand.landmarks[end];

        const startX = startLandmark.x * canvas.width;
        const startY = startLandmark.y * canvas.height;
        const endX = endLandmark.x * canvas.width;
        const endY = endLandmark.y * canvas.height;

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      });
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
      {/* Status indicators */}
      <div className="flex gap-4 text-sm">
        <div className={`flex items-center gap-2 ${isWebcamActive ? 'text-green-600' : 'text-red-600'}`}>
          <div className={`w-3 h-3 rounded-full ${isWebcamActive ? 'bg-green-500' : 'bg-red-500'}`} />
          Webcam: {isWebcamActive ? 'Active' : 'Inactive'}
        </div>
        <div className={`flex items-center gap-2 ${isModelReady ? 'text-green-600' : 'text-amber-600'}`}>
          <div className={`w-3 h-3 rounded-full ${isModelReady ? 'bg-green-500' : 'bg-amber-500'}`} />
          Gesture model: {isModelReady ? 'Ready' : 'Loading…'}
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded max-w-md text-sm">
          {error}
        </div>
      )}

      {/* Landmark data display */}
      {landmarkData && (
        <div className="text-sm text-gray-600">
          Hands detected: {landmarkData.hands.length} |
          Last update: {new Date(landmarkData.timestamp * 1000).toLocaleTimeString()}
        </div>
      )}

      {/* Video and Canvas container */}
      <div className="relative">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="max-w-full h-auto border rounded"
          style={{ maxWidth: '640px', maxHeight: '480px' }}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 pointer-events-none"
        />
      </div>
    </div>
  );
};

export default WebcamLandmarks;

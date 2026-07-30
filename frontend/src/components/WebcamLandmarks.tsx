'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';

interface Landmark {
  x: number;
  y: number;
  z: number;
}

interface HandData {
  handedness: string;
  landmarks: Landmark[];
}

interface LandmarkData {
  timestamp: number;
  hands: HandData[];
  error?: string;
}

const WebcamLandmarks: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameIntervalRef = useRef<number | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [landmarkData, setLandmarkData] = useState<LandmarkData | null>(null);

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

  // Capture the current video frame and send it to the backend as a JPEG
  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const ws = wsRef.current;
    if (!video || video.readyState < 2 || !ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }

    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas');
    }
    const canvas = captureCanvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(blob);
        }
      },
      'image/jpeg',
      0.7
    );
  }, []);

  const stopFrameLoop = useCallback(() => {
    if (frameIntervalRef.current !== null) {
      window.clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  }, []);

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const ws = new WebSocket('ws://localhost:8000/ws/landmarks');
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
        stopFrameLoop();
        frameIntervalRef.current = window.setInterval(sendFrame, 1000 / 15);
      };

      ws.onmessage = (event) => {
        try {
          const data: LandmarkData = JSON.parse(event.data);
          if (data.error) {
            setError(data.error);
          } else {
            setLandmarkData(data);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        stopFrameLoop();
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        setError('WebSocket connection failed');
        setIsConnected(false);
        stopFrameLoop();
      };

      wsRef.current = ws;
    } catch {
      setError('Failed to create WebSocket connection');
      setIsConnected(false);
    }
  }, [sendFrame, stopFrameLoop]);

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    stopFrameLoop();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setIsConnected(false);
    }
  }, [stopFrameLoop]);

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

      // Draw hand label
      if (hand.landmarks.length > 0) {
        const wrist = hand.landmarks[0];
        const labelX = wrist.x * canvas.width;
        const labelY = wrist.y * canvas.height - 20;
        
        ctx.fillStyle = color;
        ctx.font = '16px Arial';
        ctx.fillText(hand.handedness, labelX, labelY);
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

  // Auto-start webcam and connect to backend on mount
  useEffect(() => {
    startWebcam();
  }, [startWebcam]);

  useEffect(() => {
    if (isWebcamActive) {
      connectWebSocket();
    }
  }, [isWebcamActive, connectWebSocket]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
      disconnectWebSocket();
    };
  }, [stopWebcam, disconnectWebSocket]);

  return (
    <div className="flex flex-col items-center gap-4 p-4">
      <h2 className="text-2xl font-bold text-center">AirKeys Hand Tracking</h2>

      {/* Status indicators */}
      <div className="flex gap-4 text-sm">
        <div className={`flex items-center gap-2 ${isWebcamActive ? 'text-green-600' : 'text-red-600'}`}>
          <div className={`w-3 h-3 rounded-full ${isWebcamActive ? 'bg-green-500' : 'bg-red-500'}`} />
          Webcam: {isWebcamActive ? 'Active' : 'Inactive'}
        </div>
        <div className={`flex items-center gap-2 ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          Backend: {isConnected ? 'Connected' : 'Disconnected'}
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

      {/* Instructions */}
      <div className="text-sm text-gray-600 max-w-md text-center">
        <p className="mb-2">
          Webcam and backend connect automatically on page load.
        </p>
        <p>
          Hold your hand in front of the camera to see 21 tracked landmarks in real-time!
        </p>
      </div>
    </div>
  );
};

export default WebcamLandmarks;
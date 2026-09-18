'use client';

import React, { useEffect, useRef } from 'react';
import type * as Tone from 'tone';

interface AudioVisualizerProps {
  // null before audio is enabled — renders a gentle idle animation instead.
  analyser: Tone.Analyser | null;
}

const BAR_COUNT = 40;

// FFT magnitudes from Tone.Analyser come back as dB, roughly -100 (silence)
// to 0 (loud). This maps that range to 0..1 for bar height.
function dbToUnit(db: number): number {
  return Math.max(0, Math.min(1, (db + 100) / 65));
}

const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ analyser }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const idlePhaseRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      let values: number[];
      if (analyser) {
        const raw = analyser.getValue();
        const bins = Array.isArray(raw) ? raw[0] : raw;
        // Bias sampling toward the lower bins — that's where a synth voice's
        // energy actually sits, so evenly-spaced bins would look mostly flat
        // on the right half of the display.
        values = Array.from({ length: BAR_COUNT }, (_, i) => {
          const bin = Math.floor((i / BAR_COUNT) ** 1.6 * (bins.length * 0.5));
          return dbToUnit(bins[Math.min(bin, bins.length - 1)]);
        });
      } else {
        idlePhaseRef.current += 0.04;
        values = Array.from(
          { length: BAR_COUNT },
          (_, i) => 0.06 + 0.05 * Math.sin(idlePhaseRef.current + i * 0.4)
        );
      }

      const barWidth = width / BAR_COUNT;
      const gap = barWidth * 0.3;
      const radius = Math.min(3, (barWidth - gap) / 2);

      values.forEach((v, i) => {
        const barHeight = Math.max(2, v * height);
        const x0 = i * barWidth + gap / 2;
        const w = barWidth - gap;
        const y0 = height - barHeight;

        const gradient = ctx.createLinearGradient(0, height, 0, 0);
        gradient.addColorStop(0, '#26221c');
        gradient.addColorStop(1, '#6b3f4f');
        ctx.fillStyle = gradient;

        ctx.beginPath();
        ctx.moveTo(x0, height);
        ctx.lineTo(x0, y0 + radius);
        ctx.arcTo(x0, y0, x0 + radius, y0, radius);
        ctx.lineTo(x0 + w - radius, y0);
        ctx.arcTo(x0 + w, y0, x0 + w, y0 + radius, radius);
        ctx.lineTo(x0 + w, height);
        ctx.closePath();
        ctx.fill();
      });

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [analyser]);

  return <canvas ref={canvasRef} className="w-full h-16 block" />;
};

export default AudioVisualizer;

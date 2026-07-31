// Two-octave C major scale, low to high
export const SCALE = [
  'C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3',
  'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
];

const PINCH_THRESHOLD = 0.06;

export function isPinching(
  thumb: { x: number; y: number },
  indexTip: { x: number; y: number }
): boolean {
  const dx = thumb.x - indexTip.x;
  const dy = thumb.y - indexTip.y;
  return Math.hypot(dx, dy) < PINCH_THRESHOLD;
}

// Maps normalized vertical position (0 = top of frame, 1 = bottom) to a scale note.
export function noteForHeight(y: number): string {
  const index = Math.min(
    SCALE.length - 1,
    Math.max(0, Math.floor((1 - y) * SCALE.length))
  );
  return SCALE[index];
}

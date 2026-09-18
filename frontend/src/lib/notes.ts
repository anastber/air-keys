const PINCH_THRESHOLD = 0.06;

export function isPinching(
  thumb: { x: number; y: number },
  indexTip: { x: number; y: number }
): boolean {
  const dx = thumb.x - indexTip.x;
  const dy = thumb.y - indexTip.y;
  return Math.hypot(dx, dy) < PINCH_THRESHOLD;
}

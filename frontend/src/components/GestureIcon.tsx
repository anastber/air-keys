import React from 'react';

// Hand-drawn (well — hand-coded) line-art gesture icons, replacing emoji
// throughout the app. Each gesture is just a set of finger states (up or
// folded) plus a thumb position — which is literally the same feature
// space the trained classifier and the kNN work in (see lib/customGestures
// ::normalizeLandmarks) — so the icon system doubles as an honest picture
// of what the app is actually looking at, not decoration bolted on after
// the fact.

const SLOTS = [31, 41, 51, 61]; // x-centers: index, middle, ring, pinky
const EXTENDED_TOP = [16, 10, 16, 22]; // y-top when that finger is up
const FOLDED_TOP = 38; // y-top when folded (short stub)
const PALM_BOTTOM = 48;

type FingerMask = [index: boolean, middle: boolean, ring: boolean, pinky: boolean];
type ThumbState = 'out' | 'in' | 'up';

interface GestureSpec {
  fingers: FingerMask;
  thumb: ThumbState;
  // Thumb-to-index contact loop (pinch, ok_sign).
  ring?: boolean;
}

const GESTURES: Record<string, GestureSpec> = {
  fist: { fingers: [false, false, false, false], thumb: 'in' },
  point: { fingers: [true, false, false, false], thumb: 'in' },
  peace: { fingers: [true, true, false, false], thumb: 'in' },
  open_palm: { fingers: [true, true, true, true], thumb: 'out' },
  pinch: { fingers: [false, false, false, false], thumb: 'out', ring: true },
  ok_sign: { fingers: [false, true, true, true], thumb: 'out', ring: true },
  rock_on: { fingers: [true, false, false, true], thumb: 'in' },
  call_me: { fingers: [false, false, false, true], thumb: 'out' },
  thumbs_up: { fingers: [false, false, false, false], thumb: 'up' },
};

// Fallback for taught/custom gestures, which have no fixed hand shape:
// a relaxed open hand.
const DEFAULT_GESTURE: GestureSpec = { fingers: [true, true, true, true], thumb: 'out' };

function fingerRects(mask: FingerMask) {
  return SLOTS.map((x, i) => {
    const yTop = mask[i] ? EXTENDED_TOP[i] : FOLDED_TOP;
    return { x: x - 4, y: yTop, w: 8, h: PALM_BOTTOM - yTop };
  });
}

interface GestureIconProps {
  label: string;
  size?: number;
  className?: string;
}

const GestureIcon: React.FC<GestureIconProps> = ({ label, size = 22, className }) => {
  const spec = GESTURES[label] ?? DEFAULT_GESTURE;
  const fingers = fingerRects(spec.fingers);

  return (
    <svg
      width={size}
      height={(size * 58) / 70}
      viewBox="-15 -5 100 70"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      className={className}
      aria-hidden="true"
    >
      <rect x="15" y="34" width="44" height="22" rx="11" />
      {fingers.map((f, i) => (
        <rect key={i} x={f.x} y={f.y} width={f.w} height={f.h} rx={4} />
      ))}
      {spec.thumb === 'out' && (
        <rect x="0" y="0" width="9" height="24" rx="4.5" transform="translate(13,40) rotate(-55)" />
      )}
      {spec.thumb === 'in' && (
        <rect x="0" y="0" width="8" height="13" rx="4" transform="translate(12,36) rotate(-15)" />
      )}
      {spec.thumb === 'up' && <rect x="19" y="16" width="8" height="30" rx="4" />}
      {spec.ring && <circle cx="20" cy="32" r="5" />}
    </svg>
  );
};

export default GestureIcon;

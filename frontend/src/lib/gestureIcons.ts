// Emoji stand-ins for each gesture — used in the song guide, rule editor,
// and canvas overlay so a visitor recognizes a pose at a glance instead of
// reading a label.

export const GESTURE_EMOJI: Record<string, string> = {
  fist: '✊',
  open_palm: '✋',
  pinch: '🤏',
  point: '☝️',
  peace: '✌️',
  thumbs_up: '👍',
  ok_sign: '👌',
  rock_on: '🤘',
  call_me: '🤙',
  // Background/negative class for the trained classifier (see
  // lib/trainedGestures.ts) — never an actionable label in production, but
  // shown in the dataset collector so it has a recognizable icon too.
  no_gesture: '🤚',
};

// Shown for a visitor's own taught gestures, which have no fixed icon.
const DEFAULT_GESTURE_EMOJI = '🖐️';

export function gestureEmoji(label: string): string {
  return GESTURE_EMOJI[label] ?? DEFAULT_GESTURE_EMOJI;
}

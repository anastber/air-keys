// Emoji stand-ins for each gesture — used in the combo guide, rule editor,
// and canvas overlay so a visitor recognizes a pose at a glance instead of
// reading a label.

export const GESTURE_EMOJI: Record<string, string> = {
  fist: '✊',
  open_palm: '✋',
  pinch: '🤏',
  point: '☝️',
  peace: '✌️',
  thumbs_up: '👍',
};

// Shown for a visitor's own taught gestures, which have no fixed icon.
const DEFAULT_GESTURE_EMOJI = '🖐️';

export function gestureEmoji(label: string): string {
  return GESTURE_EMOJI[label] ?? DEFAULT_GESTURE_EMOJI;
}

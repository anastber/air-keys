// The control layer: maps a classified gesture to a musical action.
//
// This is deliberately separate from the classifier. The classifier's job
// is perception — "which of the 6 trained poses is this hand making" — a
// problem you genuinely can't hand-code reliably. What that pose *does*
// musically is a plain, user-editable lookup table with no ML in it at all.
// See RuleEditor for the UI that edits these live.

export type ActionType = 'note' | 'chord' | 'bass' | 'arpeggio' | 'sustain_toggle';
export type Voicing = 'close' | 'open';

export interface OctaveRange {
  min: number;
  max: number;
}

export interface GestureRule {
  action: ActionType;
  voicing?: Voicing; // only meaningful for 'chord'
  octaveRange: OctaveRange; // ignored for 'sustain_toggle'
}

export interface MusicalAction {
  type: ActionType;
  notes: string[]; // 1 note for note/bass, 3 for chord/arpeggio, 0 for sustain_toggle
}

export const DEFAULT_RULES: Record<string, GestureRule> = {
  fist: { action: 'bass', octaveRange: { min: 2, max: 2 } },
  open_palm: { action: 'chord', voicing: 'close', octaveRange: { min: 3, max: 4 } },
  pinch: { action: 'note', octaveRange: { min: 3, max: 5 } },
  point: { action: 'arpeggio', octaveRange: { min: 4, max: 5 } },
  peace: { action: 'chord', voicing: 'open', octaveRange: { min: 3, max: 4 } },
  thumbs_up: { action: 'sustain_toggle', octaveRange: { min: 3, max: 3 } },
};

// --- Music theory helpers -------------------------------------------------
// Everything below works in diatonic scale degrees rather than raw
// semitones, so "third" and "fifth" always land on the right note of a C
// major scale regardless of octave.

const MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]; // semitones from root, per degree
const NOTE_NAMES = [
  'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
];

function midiToNoteName(midi: number): string {
  const octave = Math.floor(midi / 12) - 1;
  const name = NOTE_NAMES[((midi % 12) + 12) % 12];
  return `${name}${octave}`;
}

function noteNameToMidi(note: string): number | null {
  const match = note.match(/^([A-G]#?)(-?\d+)$/);
  if (!match) return null;
  const [, name, octaveStr] = match;
  const index = NOTE_NAMES.indexOf(name);
  return (parseInt(octaveStr, 10) + 1) * 12 + index;
}

function transposeNote(note: string, semitones: number): string {
  const midi = noteNameToMidi(note);
  return midi === null ? note : midiToNoteName(midi + semitones);
}

// degree 0 = root of baseOctave, degree 7 = the same root one octave up.
// Negative degrees and degrees beyond a single octave both wrap correctly.
function noteAtDegree(degree: number, baseOctave: number): string {
  const octaveShift = Math.floor(degree / 7);
  const stepInOctave = ((degree % 7) + 7) % 7;
  const semitone = MAJOR_SCALE_INTERVALS[stepInOctave];
  const midi = (baseOctave + octaveShift + 1) * 12 + semitone;
  return midiToNoteName(midi);
}

// Maps normalized hand height (0 = top of frame, 1 = bottom) to a scale
// degree spanning the given octave range.
function degreeForHeight(y: number, range: OctaveRange): number {
  const totalDegrees = (range.max - range.min + 1) * 7;
  return Math.min(totalDegrees - 1, Math.max(0, Math.floor((1 - y) * totalDegrees)));
}

function noteForHeight(y: number, range: OctaveRange): string {
  return noteAtDegree(degreeForHeight(y, range), range.min);
}

// Diatonic triad (root + stacked thirds) built on the scale degree at the
// given height. 'open' voicing spreads the third up an octave for a wider,
// less clustered sound.
function triadForHeight(y: number, range: OctaveRange, voicing: Voicing): string[] {
  const degree = degreeForHeight(y, range);
  const root = noteAtDegree(degree, range.min);
  const third = noteAtDegree(degree + 2, range.min);
  const fifth = noteAtDegree(degree + 4, range.min);
  return voicing === 'open' ? [root, fifth, transposeNote(third, 12)] : [root, third, fifth];
}

// --- Rule resolution -------------------------------------------------------

/** Turn a rule + current hand height into the concrete notes to play. */
export function resolveAction(rule: GestureRule, height: number): MusicalAction {
  switch (rule.action) {
    case 'note':
    case 'bass':
      return { type: rule.action, notes: [noteForHeight(height, rule.octaveRange)] };
    case 'chord':
      return {
        type: 'chord',
        notes: triadForHeight(height, rule.octaveRange, rule.voicing ?? 'close'),
      };
    case 'arpeggio':
      return { type: 'arpeggio', notes: triadForHeight(height, rule.octaveRange, 'close') };
    case 'sustain_toggle':
      return { type: 'sustain_toggle', notes: [] };
  }
}

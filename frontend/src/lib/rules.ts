// The control layer: maps a classified gesture to a musical action.
//
// This is deliberately separate from the classifiers (MediaPipe's pretrained
// model, the self-trained MLP, and the client-side kNN). Their job is
// perception — "which gesture is this hand making" — a problem you
// genuinely can't hand-code reliably. What that gesture *does* musically is
// a plain, user-editable lookup table with no ML in it at all.
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
  // Fixed scale degree (0 = do, 6 = ti, 7 = do an octave up) this gesture
  // always plays, regardless of hand height — see "solfège keyboard" below.
  // Omitted for taught/custom gestures, which keep the original continuous
  // height-to-pitch mapping (pitchInputForHeight) since they aren't part of
  // the fixed 8-note layout.
  degree?: number;
}

export interface MusicalAction {
  type: ActionType;
  notes: string[]; // 1 note for note/bass, 3 for chord/arpeggio, 0 for sustain_toggle
}

// Always available, zero setup: 'pinch' is pure geometry (lib/notes.ts),
// the rest come from MediaPipe's pretrained recognizer (lib/gestureRecognition.ts).
// A visitor's taught gestures (lib/customGestures.ts) are added on top of this list.
export const BASE_GESTURE_LABELS = ['fist', 'open_palm', 'pinch', 'point', 'peace', 'thumbs_up'];

// Poses MediaPipe's pretrained recognizer doesn't cover, classified instead
// by a small MLP trained offline on a dataset collected for this project
// (see ml/train.py and lib/trainedGestures.ts). Also always available, zero
// setup — the model ships with the app rather than being taught per-visitor.
export const TRAINED_GESTURE_LABELS = ['ok_sign', 'rock_on', 'call_me'];

// Fallback assigned to a newly taught custom gesture until the visitor edits it.
// No fixed `degree` — taught gestures keep the original continuous
// height-to-pitch mapping across the octave range.
export const DEFAULT_CUSTOM_RULE: GestureRule = {
  action: 'note',
  octaveRange: { min: 3, max: 5 },
};

// Solfège syllable for each fixed degree used below (do..ti, then do an
// octave up) — shown in RuleEditor and SongGuide so the mapping is
// learnable at a glance instead of memorized blind.
export const SOLFEGE_SYLLABLES = ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti', "do'"];

// The 8 always-available non-sustain gestures (6 base + 2 of the 3 trained
// — thumbs_up stays a sustain toggle) are laid out as a one-octave diatonic
// "keyboard": each gesture always plays the same scale degree, regardless
// of hand height. Earlier versions mapped hand height continuously to
// pitch, which made hitting a *specific* target note (i.e. reproducing an
// actual melody) unreliable — great for ambient soundscapes, unplayable
// for "Twinkle Twinkle". Fixing the note to the gesture and leaving height
// to (optionally) shift the octave — widen a gesture's octaveRange in the
// Rules tab to bring that back — makes a known tune reproducible: hold
// your hand steady and just cycle through gesture shapes. See SongGuide
// for ready-made gesture sequences that only need this one octave.
export const DEFAULT_RULES: Record<string, GestureRule> = {
  fist: { action: 'note', degree: 0, octaveRange: { min: 4, max: 4 } }, // do (C4)
  point: { action: 'note', degree: 1, octaveRange: { min: 4, max: 4 } }, // re (D4)
  peace: { action: 'note', degree: 2, octaveRange: { min: 4, max: 4 } }, // mi (E4)
  open_palm: { action: 'note', degree: 3, octaveRange: { min: 4, max: 4 } }, // fa (F4)
  pinch: { action: 'note', degree: 4, octaveRange: { min: 4, max: 4 } }, // sol (G4)
  ok_sign: { action: 'note', degree: 5, octaveRange: { min: 4, max: 4 } }, // la (A4)
  rock_on: { action: 'note', degree: 6, octaveRange: { min: 4, max: 4 } }, // ti (B4)
  call_me: { action: 'note', degree: 7, octaveRange: { min: 4, max: 4 } }, // do' (C5)
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

// Resolves a rule + hand height down to (scale degree, base octave):
//   - fixed-degree gestures (rule.degree set): the degree never changes:
//     height only picks which octave band within octaveRange to play it
//     in (1 band = height has no effect at all, the default — see above).
//   - taught/custom gestures (no rule.degree): the original continuous
//     mapping, height sweeping the full degree range across octaveRange.
function pitchInputForHeight(
  rule: GestureRule,
  y: number
): { degree: number; baseOctave: number } {
  const numBands = rule.octaveRange.max - rule.octaveRange.min + 1;
  if (rule.degree !== undefined) {
    const band = Math.min(numBands - 1, Math.max(0, Math.floor((1 - y) * numBands)));
    return { degree: rule.degree, baseOctave: rule.octaveRange.min + band };
  }
  const totalDegrees = numBands * 7;
  const degree = Math.min(totalDegrees - 1, Math.max(0, Math.floor((1 - y) * totalDegrees)));
  return { degree, baseOctave: rule.octaveRange.min };
}

function noteForHeight(rule: GestureRule, y: number): string {
  const { degree, baseOctave } = pitchInputForHeight(rule, y);
  return noteAtDegree(degree, baseOctave);
}

// Diatonic triad (root + stacked thirds) built on the rule's resolved scale
// degree. 'open' voicing spreads the third up an octave for a wider, less
// clustered sound.
function triadForHeight(rule: GestureRule, y: number, voicing: Voicing): string[] {
  const { degree, baseOctave } = pitchInputForHeight(rule, y);
  const root = noteAtDegree(degree, baseOctave);
  const third = noteAtDegree(degree + 2, baseOctave);
  const fifth = noteAtDegree(degree + 4, baseOctave);
  return voicing === 'open' ? [root, fifth, transposeNote(third, 12)] : [root, third, fifth];
}

// --- Rule resolution -------------------------------------------------------

/** Turn a rule + current hand height into the concrete notes to play. */
export function resolveAction(rule: GestureRule, height: number): MusicalAction {
  switch (rule.action) {
    case 'note':
    case 'bass':
      return { type: rule.action, notes: [noteForHeight(rule, height)] };
    case 'chord':
      return { type: 'chord', notes: triadForHeight(rule, height, rule.voicing ?? 'close') };
    case 'arpeggio':
      return { type: 'arpeggio', notes: triadForHeight(rule, height, 'close') };
    case 'sustain_toggle':
      return { type: 'sustain_toggle', notes: [] };
  }
}

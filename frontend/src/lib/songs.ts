// Ready-made gesture sequences for a few famous, entirely diatonic melodies
// — chosen specifically because they only use scale degrees 1-6 (do..la)
// and stay within one octave, so they play back exactly against the fixed
// gesture layout in rules.ts::DEFAULT_RULES with zero hand-height movement
// required. Purely a reference guide (see SongGuide) — doesn't drive audio.

export interface Song {
  name: string;
  // Each inner array is one musical phrase, rendered on its own line.
  phrases: string[][];
}

export const SONGS: Song[] = [
  {
    name: 'Twinkle Twinkle Little Star',
    phrases: [
      ['fist', 'fist', 'pinch', 'pinch', 'ok_sign', 'ok_sign', 'pinch'],
      ['open_palm', 'open_palm', 'peace', 'peace', 'point', 'point', 'fist'],
    ],
  },
  {
    name: 'Ode to Joy',
    phrases: [
      ['peace', 'peace', 'open_palm', 'pinch'],
      ['pinch', 'open_palm', 'peace', 'point'],
      ['fist', 'fist', 'point', 'peace'],
      ['peace', 'point', 'point'],
    ],
  },
  {
    name: 'Mary Had a Little Lamb',
    phrases: [
      ['peace', 'point', 'fist', 'point'],
      ['peace', 'peace', 'peace'],
      ['point', 'point', 'point'],
      ['peace', 'pinch', 'pinch'],
      ['peace', 'point', 'fist', 'point'],
      ['peace', 'peace', 'peace', 'peace'],
      ['point', 'point', 'peace', 'point'],
      ['fist'],
    ],
  },
];

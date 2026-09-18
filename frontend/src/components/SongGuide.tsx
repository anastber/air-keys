'use client';

import React from 'react';
import type { Song } from '@/lib/songs';
import { gestureEmoji } from '@/lib/gestureIcons';

interface SongGuideProps {
  songs: Song[];
}

const SongGuide: React.FC<SongGuideProps> = ({ songs }) => (
  <div className="flex flex-col gap-4">
    <p className="text-sm text-ak-muted">
      Each gesture below always plays the same note (see the Rules tab) — hold your
      hand at a steady height and just cycle through the shapes in order.
    </p>
    <div className="flex flex-col gap-4">
      {songs.map((song) => (
        <div
          key={song.name}
          className="flex flex-col gap-3 rounded-lg bg-ak-panel/60 border border-ak-border px-4 py-3"
        >
          <span className="text-sm font-display italic text-ak-accent">{song.name}</span>
          <div className="flex flex-col gap-2">
            {song.phrases.map((phrase, i) => (
              <div key={i} className="flex items-center gap-2 flex-wrap">
                {phrase.map((gesture, j) => (
                  <React.Fragment key={j}>
                    <span title={gesture} className="text-2xl leading-none">
                      {gestureEmoji(gesture)}
                    </span>
                    {j < phrase.length - 1 && <span className="text-ak-subtle text-xs">→</span>}
                  </React.Fragment>
                ))}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  </div>
);

export default SongGuide;

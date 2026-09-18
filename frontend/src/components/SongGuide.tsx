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
          className="flex flex-col gap-2 rounded-xl bg-white/[0.03] border border-white/[0.06] px-3 py-2.5"
        >
          <span className="text-sm font-semibold text-ak-violet">{song.name}</span>
          <div className="flex flex-col gap-1.5">
            {song.phrases.map((phrase, i) => (
              <div key={i} className="flex items-center gap-1.5 text-xl leading-none flex-wrap">
                {phrase.map((gesture, j) => (
                  <React.Fragment key={j}>
                    <span title={gesture}>{gestureEmoji(gesture)}</span>
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

'use client';

import React from 'react';
import type { Combo } from '@/lib/rules';
import { gestureEmoji } from '@/lib/gestureIcons';

interface ComboGuideProps {
  combos: Combo[];
}

const ComboGuide: React.FC<ComboGuideProps> = ({ combos }) => (
  <div className="ak-glass rounded-2xl p-5 w-full max-w-sm flex flex-col gap-3">
    <div className="flex items-center gap-2">
      <span className="text-lg">🎶</span>
      <h3 className="font-semibold text-ak-text">Try These Combos</h3>
    </div>
    <p className="text-sm text-ak-muted">
      Do 3 gestures in order, either hand, within a few seconds — each still plays
      normally, plus a bonus tune on the third.
    </p>
    <div className="flex flex-col gap-2">
      {combos.map((combo) => (
        <div
          key={combo.name}
          className="flex items-center gap-3 rounded-xl bg-white/[0.03] border border-white/[0.06] px-3 py-2.5 hover:border-ak-violet/40 transition-colors"
        >
          <span className="text-sm font-semibold text-ak-violet w-16 shrink-0">{combo.name}</span>
          <span className="flex items-center gap-1.5 text-xl leading-none">
            {combo.sequence.map((gesture, i) => (
              <React.Fragment key={i}>
                <span title={gesture}>{gestureEmoji(gesture)}</span>
                {i < combo.sequence.length - 1 && (
                  <span className="text-ak-subtle text-xs">→</span>
                )}
              </React.Fragment>
            ))}
          </span>
        </div>
      ))}
    </div>
  </div>
);

export default ComboGuide;

'use client';

import React, { useState } from 'react';
import ComboGuide from '@/components/ComboGuide';
import GestureTrainer from '@/components/GestureTrainer';
import RuleEditor from '@/components/RuleEditor';
import SongGuide from '@/components/SongGuide';
import type { Combo, GestureRule } from '@/lib/rules';
import type { Song } from '@/lib/songs';
import type { HandData } from '@/lib/types';

type TabId = 'songs' | 'combos' | 'teach' | 'rules';

const TABS: { id: TabId; label: string; icon: string }[] = [
  { id: 'songs', label: 'Songs', icon: '🎵' },
  { id: 'combos', label: 'Combos', icon: '🎶' },
  { id: 'teach', label: 'Teach', icon: '🧠' },
  { id: 'rules', label: 'Rules', icon: '🎛️' },
];

interface ControlDeckProps {
  songs: Song[];
  combos: Combo[];
  currentHand: HandData | null;
  onGestureRecorded: (label: string) => void;
  labels: string[];
  rules: Record<string, GestureRule>;
  onRuleChange: (label: string, patch: Partial<GestureRule>) => void;
}

// One shell, four panels. All four stay mounted the whole time — only
// visibility toggles — specifically so switching away from "Teach" mid
// recording doesn't reset that recording's progress.
const ControlDeck: React.FC<ControlDeckProps> = ({
  songs,
  combos,
  currentHand,
  onGestureRecorded,
  labels,
  rules,
  onRuleChange,
}) => {
  const [active, setActive] = useState<TabId>('songs');

  return (
    <div className="ak-glass rounded-2xl w-full flex flex-col overflow-hidden">
      <div className="flex border-b border-white/[0.06]">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActive(tab.id)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-3 text-sm font-medium transition-colors ${
              active === tab.id
                ? 'text-ak-text bg-white/[0.04] border-b-2 border-ak-violet'
                : 'text-ak-subtle hover:text-ak-muted border-b-2 border-transparent'
            }`}
          >
            <span>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      <div className="p-5">
        <div className={active === 'songs' ? '' : 'hidden'}>
          <SongGuide songs={songs} />
        </div>
        <div className={active === 'combos' ? '' : 'hidden'}>
          <ComboGuide combos={combos} />
        </div>
        <div className={active === 'teach' ? '' : 'hidden'}>
          <GestureTrainer currentHand={currentHand} onGestureRecorded={onGestureRecorded} />
        </div>
        <div className={active === 'rules' ? '' : 'hidden'}>
          <RuleEditor labels={labels} rules={rules} onChange={onRuleChange} />
        </div>
      </div>
    </div>
  );
};

export default ControlDeck;

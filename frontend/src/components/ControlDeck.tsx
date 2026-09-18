'use client';

import React, { useState } from 'react';
import GestureTrainer from '@/components/GestureTrainer';
import RuleEditor from '@/components/RuleEditor';
import SongGuide from '@/components/SongGuide';
import type { GestureRule } from '@/lib/rules';
import type { Song } from '@/lib/songs';
import type { HandData } from '@/lib/types';

type TabId = 'songs' | 'teach' | 'rules';

const TABS: { id: TabId; label: string }[] = [
  { id: 'songs', label: 'Songs' },
  { id: 'teach', label: 'Teach' },
  { id: 'rules', label: 'Rules' },
];

interface ControlDeckProps {
  songs: Song[];
  currentHand: HandData | null;
  onGestureRecorded: (label: string) => void;
  labels: string[];
  rules: Record<string, GestureRule>;
  onRuleChange: (label: string, patch: Partial<GestureRule>) => void;
}

// One shell, three panels. All three stay mounted the whole time — only
// visibility toggles — specifically so switching away from "Teach" mid
// recording doesn't reset that recording's progress.
const ControlDeck: React.FC<ControlDeckProps> = ({
  songs,
  currentHand,
  onGestureRecorded,
  labels,
  rules,
  onRuleChange,
}) => {
  const [active, setActive] = useState<TabId>('songs');

  return (
    <div className="ak-glass rounded-lg w-full flex flex-col overflow-hidden">
      <div className="flex border-b border-ak-border">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActive(tab.id)}
            className={`flex-1 flex items-center justify-center py-3 text-sm transition-colors border-b-2 ${
              active === tab.id
                ? 'text-ak-text border-ak-line font-medium'
                : 'text-ak-subtle hover:text-ak-muted border-transparent'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="p-5">
        <div className={active === 'songs' ? '' : 'hidden'}>
          <SongGuide songs={songs} />
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

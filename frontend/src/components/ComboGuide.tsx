'use client';

import React from 'react';
import type { Combo } from '@/lib/rules';

interface ComboGuideProps {
  combos: Combo[];
}

const ComboGuide: React.FC<ComboGuideProps> = ({ combos }) => (
  <div className="border rounded p-4 w-full max-w-md">
    <h3 className="font-bold text-lg mb-1">Try These Combos</h3>
    <p className="text-sm text-gray-600 mb-3">
      Do these 3 gestures in order, either hand, within a few seconds — each still plays
      normally, plus a little bonus tune on the third.
    </p>
    <div className="flex flex-col gap-2 text-sm">
      {combos.map((combo) => (
        <div key={combo.name} className="flex items-center gap-2">
          <span className="font-semibold w-16 shrink-0">{combo.name}</span>
          <span className="font-mono text-gray-700">{combo.sequence.join(' → ')}</span>
        </div>
      ))}
    </div>
  </div>
);

export default ComboGuide;

'use client';

import React from 'react';
import type { ActionType, GestureRule, Voicing } from '@/lib/rules';

interface RuleEditorProps {
  labels: string[];
  rules: Record<string, GestureRule>;
  onChange: (label: string, patch: Partial<GestureRule>) => void;
}

const ACTIONS: ActionType[] = ['note', 'chord', 'bass', 'arpeggio', 'sustain_toggle'];
const VOICINGS: Voicing[] = ['close', 'open'];

const RuleEditor: React.FC<RuleEditorProps> = ({ labels, rules, onChange }) => (
  <div className="border rounded p-4 w-full max-w-md">
    <h3 className="font-bold text-lg mb-1">Gesture Rules</h3>
    <p className="text-sm text-gray-600 mb-3">
      What each gesture plays. Edit live — a change applies the next time you make
      that gesture, no retraining needed.
    </p>
    <div className="flex flex-col gap-2">
      {labels.map((label) => {
        const rule = rules[label];
        if (!rule) return null;
        return (
          <div key={label} className="flex flex-wrap items-center gap-2 text-sm border-b pb-2">
            <span className="font-mono w-24 shrink-0">{label}</span>
            <select
              value={rule.action}
              onChange={(e) => onChange(label, { action: e.target.value as ActionType })}
              className="border rounded px-1 py-0.5"
            >
              {ACTIONS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
            {rule.action === 'chord' && (
              <select
                value={rule.voicing ?? 'close'}
                onChange={(e) => onChange(label, { voicing: e.target.value as Voicing })}
                className="border rounded px-1 py-0.5"
              >
                {VOICINGS.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            )}
            {rule.action !== 'sustain_toggle' && (
              <span className="flex items-center gap-1 text-xs text-gray-600">
                octaves
                <input
                  type="number"
                  min={0}
                  max={8}
                  value={rule.octaveRange.min}
                  onChange={(e) =>
                    onChange(label, {
                      octaveRange: { ...rule.octaveRange, min: Number(e.target.value) },
                    })
                  }
                  className="border rounded w-12 px-1"
                />
                –
                <input
                  type="number"
                  min={0}
                  max={8}
                  value={rule.octaveRange.max}
                  onChange={(e) =>
                    onChange(label, {
                      octaveRange: { ...rule.octaveRange, max: Number(e.target.value) },
                    })
                  }
                  className="border rounded w-12 px-1"
                />
              </span>
            )}
          </div>
        );
      })}
    </div>
  </div>
);

export default RuleEditor;

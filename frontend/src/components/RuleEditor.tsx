'use client';

import React from 'react';
import type { ActionType, GestureRule, Voicing } from '@/lib/rules';
import { gestureEmoji } from '@/lib/gestureIcons';

interface RuleEditorProps {
  labels: string[];
  rules: Record<string, GestureRule>;
  onChange: (label: string, patch: Partial<GestureRule>) => void;
}

const ACTIONS: ActionType[] = ['note', 'chord', 'bass', 'arpeggio', 'sustain_toggle'];
const VOICINGS: Voicing[] = ['close', 'open'];

const selectClass =
  'bg-white/[0.06] border border-white/10 rounded-lg px-2 py-1 text-ak-text text-xs focus:outline-none focus:ring-1 focus:ring-ak-violet/60 focus:border-ak-violet/60';
const numberClass =
  'bg-white/[0.06] border border-white/10 rounded-lg w-12 px-1.5 py-1 text-ak-text text-xs text-center focus:outline-none focus:ring-1 focus:ring-ak-violet/60 focus:border-ak-violet/60';

const RuleEditor: React.FC<RuleEditorProps> = ({ labels, rules, onChange }) => (
  <div className="ak-glass rounded-2xl p-5 w-full max-w-sm flex flex-col gap-3">
    <div className="flex items-center gap-2">
      <span className="text-lg">🎛️</span>
      <h3 className="font-semibold text-ak-text">Gesture Rules</h3>
    </div>
    <p className="text-sm text-ak-muted">
      What each gesture plays. Edit live — a change applies the next time you make
      that gesture, no retraining needed.
    </p>
    <div className="flex flex-col gap-2">
      {labels.map((label) => {
        const rule = rules[label];
        if (!rule) return null;
        return (
          <div
            key={label}
            className="flex flex-wrap items-center gap-2 rounded-xl bg-white/[0.03] border border-white/[0.06] px-3 py-2.5"
          >
            <span className="flex items-center gap-1.5 w-24 shrink-0">
              <span className="text-base leading-none">{gestureEmoji(label)}</span>
              <span className="font-mono text-xs text-ak-muted truncate">{label}</span>
            </span>
            <select
              value={rule.action}
              onChange={(e) => onChange(label, { action: e.target.value as ActionType })}
              className={selectClass}
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
                className={selectClass}
              >
                {VOICINGS.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            )}
            {rule.action !== 'sustain_toggle' && (
              <span className="flex items-center gap-1 text-xs text-ak-subtle">
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
                  className={numberClass}
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
                  className={numberClass}
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

'use client';

import React from 'react';
import { SOLFEGE_SYLLABLES, type ActionType, type GestureRule, type Voicing } from '@/lib/rules';
import { gestureEmoji } from '@/lib/gestureIcons';

interface RuleEditorProps {
  labels: string[];
  rules: Record<string, GestureRule>;
  onChange: (label: string, patch: Partial<GestureRule>) => void;
}

const ACTIONS: ActionType[] = ['note', 'chord', 'bass', 'arpeggio', 'sustain_toggle'];
const VOICINGS: Voicing[] = ['close', 'open'];

const selectClass =
  'bg-transparent border border-ak-border rounded-md px-2 py-1 text-ak-text text-xs focus:outline-none focus:ring-1 focus:ring-ak-accent/60 focus:border-ak-accent/60';
const numberClass =
  'bg-transparent border border-ak-border rounded-md w-12 px-1.5 py-1 text-ak-text text-xs text-center focus:outline-none focus:ring-1 focus:ring-ak-accent/60 focus:border-ak-accent/60';

const RuleEditor: React.FC<RuleEditorProps> = ({ labels, rules, onChange }) => (
  <div className="flex flex-col gap-3">
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
            className="flex flex-wrap items-center gap-2 rounded-lg bg-ak-panel/60 border border-ak-border px-3 py-2.5"
          >
            <span className="flex items-center gap-2 w-28 shrink-0">
              <span className="text-lg leading-none shrink-0">{gestureEmoji(label)}</span>
              <span className="flex flex-col leading-tight min-w-0">
                <span className="text-xs text-ak-muted truncate">{label}</span>
                {rule.degree !== undefined && (
                  <span className="text-[10px] text-ak-accent">
                    {SOLFEGE_SYLLABLES[rule.degree]}
                  </span>
                )}
              </span>
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

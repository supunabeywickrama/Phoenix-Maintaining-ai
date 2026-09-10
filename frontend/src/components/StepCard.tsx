"use client";

import { CheckCircle2, HelpCircle, Wrench } from "lucide-react";

interface StepCardProps {
  onAction: (message: string, mode: "answer" | "wizard") => void;
  disabled?: boolean;
  /** Shown only under the most recent agent reply. */
  visible: boolean;
  /** The wizard is already running, so offer progress actions instead of starting it. */
  inWizard: boolean;
}

/**
 * Quick replies under the latest answer.
 *
 * These send ordinary messages rather than a bespoke protocol — the retrieval
 * prompt already adapts to "done" versus "stuck" from the conversation history.
 */
export default function StepCard({ onAction, disabled, visible, inWizard }: StepCardProps) {
  if (!visible) return null;

  const actions = inWizard
    ? [
        {
          label: "Done — next step",
          icon: CheckCircle2,
          tone: "emerald",
          message: "I have completed that step. What is the next one?",
          mode: "wizard" as const,
        },
        {
          label: "I'm stuck",
          icon: HelpCircle,
          tone: "amber",
          message: "I am stuck on that step. Explain it in simpler detail and show any diagram.",
          mode: "wizard" as const,
        },
      ]
    : [
        {
          label: "Guide me step by step",
          icon: Wrench,
          tone: "orange",
          message: "Walk me through fixing this, one step at a time.",
          mode: "wizard" as const,
        },
      ];

  const tones: Record<string, string> = {
    emerald: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20",
    amber: "border-amber-500/30 bg-amber-500/10 text-amber-400 hover:bg-amber-500/20",
    orange: "border-orange-500/30 bg-orange-500/10 text-orange-400 hover:bg-orange-500/20",
  };

  return (
    <div className="flex flex-wrap gap-2 pl-11">
      {actions.map((a) => {
        const Icon = a.icon;
        return (
          <button
            key={a.label}
            disabled={disabled}
            onClick={() => onAction(a.message, a.mode)}
            className={`flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-bold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${tones[a.tone]}`}
          >
            <Icon size={14} /> {a.label}
          </button>
        );
      })}
    </div>
  );
}

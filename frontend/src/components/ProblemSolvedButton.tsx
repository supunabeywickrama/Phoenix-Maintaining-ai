"use client";

import { CheckCircle2, PartyPopper } from "lucide-react";

interface ProblemSolvedButtonProps {
  resolved: boolean;
  /** Glow once the guided fix is under way — that is when "solved" becomes likely. */
  active: boolean;
  onClick: () => void;
}

/**
 * Header action for closing out a fault. Before saving it invites the click;
 * after saving it becomes a quiet badge that still reopens the record for edits.
 */
export default function ProblemSolvedButton({ resolved, active, onClick }: ProblemSolvedButtonProps) {
  if (resolved) {
    return (
      <button
        onClick={onClick}
        title="Fix recorded — click to edit the record"
        className="flex items-center gap-1.5 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs font-bold text-emerald-400 hover:bg-emerald-500/20"
      >
        <CheckCircle2 size={14} className="motion-safe:animate-solved-pop" /> Solved
      </button>
    );
  }

  return (
    <button
      onClick={onClick}
      className={`group flex items-center gap-1.5 rounded-xl bg-emerald-600 px-3 py-2 text-xs font-bold text-white shadow-lg shadow-emerald-900/30 transition hover:-translate-y-px hover:bg-emerald-500 active:translate-y-0 ${
        active ? "motion-safe:animate-solved-glow" : ""
      }`}
    >
      <PartyPopper size={14} className="transition-transform group-hover:rotate-12" />
      Problem solved
    </button>
  );
}

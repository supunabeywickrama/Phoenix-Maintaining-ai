"use client";

import { MessageCircleQuestion, PencilLine, PartyPopper } from "lucide-react";
import type { PendingQuestion } from "../store/slices/assistantSlice";

const SOLVED = /^problem solved/i;

interface QuickReplyCardProps {
  questions: PendingQuestion[];
  disabled?: boolean;
  /** Sends the tapped answer as the technician's next message. */
  onAnswer: (message: string) => void;
  /** "Other…": put the question in the input box for a typed answer. */
  onOther: (prefill: string) => void;
  /** The "Problem solved" answer opens the fix record instead of sending. */
  onSolved: () => void;
}

/**
 * A question the assistant needs answered before it can go on — a reading, what
 * a check showed, whether a step fixed it — with answers the technician can tap.
 * Typing a reply works just as well; the backend treats either as the answer.
 */
export default function QuickReplyCard({
  questions,
  disabled,
  onAnswer,
  onOther,
  onSolved,
}: QuickReplyCardProps) {
  if (!questions.length) return null;

  return (
    <div className="space-y-2 pl-11">
      {questions.map((q) => (
        <div
          key={q.question}
          className="rounded-xl border border-sky-500/25 bg-sky-500/5 p-3"
        >
          <p className="mb-2 flex items-start gap-2 text-sm font-bold text-white">
            <MessageCircleQuestion size={16} className="mt-0.5 shrink-0 text-sky-400" />
            {q.question}
          </p>
          <div className="flex flex-wrap gap-2">
            {q.options.map((opt) => {
              const solved = SOLVED.test(opt);
              return (
                <button
                  key={opt}
                  disabled={disabled}
                  onClick={() => (solved ? onSolved() : onAnswer(`${q.question} — ${opt}`))}
                  className={`flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-bold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                    solved
                      ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/25"
                      : "border-sky-500/30 bg-sky-500/10 text-sky-300 hover:bg-sky-500/20"
                  }`}
                >
                  {solved && <PartyPopper size={13} />}
                  {opt}
                </button>
              );
            })}
            <button
              disabled={disabled}
              onClick={() => onOther(`${q.question} — `)}
              className="flex items-center gap-1.5 rounded-xl border border-slate-700 px-3 py-2 text-xs font-bold text-slate-400 transition-colors hover:border-slate-500 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
            >
              <PencilLine size={13} /> Other…
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

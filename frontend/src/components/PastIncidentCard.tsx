"use client";

import { useState } from "react";
import { History, User, Wrench, Hand, Bot, ChevronDown } from "lucide-react";
import type { FixMethod, PastIncident } from "../store/slices/assistantSlice";

const METHOD: Record<FixMethod, { label: string; icon: typeof Hand }> = {
  hands_on: { label: "Fixed by hand", icon: Hand },
  system_guided: { label: "Followed system instructions", icon: Bot },
  both: { label: "By hand + system instructions", icon: Wrench },
};

function formatDate(value: string | null): string {
  if (!value) return "Date not recorded";
  const d = new Date(value.replace(" ", "T"));
  return Number.isNaN(d.getTime())
    ? value
    : d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

/**
 * "This happened before on this machine" — confirmed fixes pulled from the
 * maintenance database, shown above the answer. Rendered from the stored
 * records themselves, never from the model's wording, so nothing here can be
 * invented.
 */
export default function PastIncidentCard({ incidents }: { incidents: PastIncident[] }) {
  const [expanded, setExpanded] = useState(false);
  if (!incidents.length) return null;

  const shown = expanded ? incidents : incidents.slice(0, 1);

  return (
    <section
      aria-label="Past fixes on this machine"
      className="mb-3 overflow-hidden rounded-xl border border-amber-500/30 bg-amber-500/5"
    >
      <header className="flex items-center gap-2 border-b border-amber-500/20 px-3 py-2">
        <History size={14} className="text-amber-400" />
        <span className="text-[11px] font-black uppercase tracking-wider text-amber-400">
          Happened before on this machine
        </span>
        <span className="ml-auto text-[11px] text-slate-500">
          {incidents.length} matching {incidents.length === 1 ? "fix" : "fixes"} on record
        </span>
      </header>

      <div className="divide-y divide-amber-500/10">
        {shown.map((inc, i) => {
          const method = inc.method ? METHOD[inc.method] : null;
          const MethodIcon = method?.icon;
          return (
            <article key={`${inc.session_id ?? "x"}-${i}`} className="space-y-2 px-3 py-3 text-xs">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-slate-400">
                <span className="font-bold text-slate-200">{formatDate(inc.date)}</span>
                {inc.engineer && (
                  <span className="flex items-center gap-1">
                    <User size={12} /> {inc.engineer}
                  </span>
                )}
                {method && MethodIcon && (
                  <span className="flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold text-emerald-400">
                    <MethodIcon size={11} /> {method.label}
                  </span>
                )}
              </div>

              <dl className="grid gap-1.5 sm:grid-cols-[7.5rem_1fr]">
                {inc.symptom && (
                  <>
                    <dt className="font-bold text-slate-500">Reported</dt>
                    <dd className="text-slate-300">{inc.symptom}</dd>
                  </>
                )}
                {inc.root_cause && (
                  <>
                    <dt className="font-bold text-slate-500">Actual cause</dt>
                    <dd className="text-slate-200">{inc.root_cause}</dd>
                  </>
                )}
                {inc.actions && (
                  <>
                    <dt className="font-bold text-slate-500">What fixed it</dt>
                    <dd className="whitespace-pre-line text-slate-200">{inc.actions}</dd>
                  </>
                )}
                {inc.parts_replaced && (
                  <>
                    <dt className="font-bold text-slate-500">Parts replaced</dt>
                    <dd className="text-slate-300">{inc.parts_replaced}</dd>
                  </>
                )}
                {!inc.root_cause && !inc.actions && inc.summary && (
                  <>
                    <dt className="font-bold text-slate-500">Summary</dt>
                    <dd className="text-slate-300">{inc.summary}</dd>
                  </>
                )}
              </dl>
            </article>
          );
        })}
      </div>

      {incidents.length > 1 && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex w-full items-center justify-center gap-1 border-t border-amber-500/20 py-1.5 text-[11px] font-bold text-amber-400 hover:bg-amber-500/10"
        >
          <ChevronDown size={12} className={expanded ? "rotate-180" : ""} />
          {expanded ? "Show less" : `Show ${incidents.length - 1} more`}
        </button>
      )}
    </section>
  );
}

"use client";

import { Wrench, GraduationCap } from "lucide-react";
import { ChatIntent } from "../store/slices/assistantSlice";

/**
 * Asked once, before the first message of a conversation.
 *
 * A fault on the floor and a training question want different answers out of the
 * same manual — one needs a diagnosis and a safe repair path, the other needs an
 * explanation. Inferring that from the wording was unreliable ("why does the pump
 * cavitate?" reads like both), so the technician says which up front and the whole
 * thread is steered by it.
 */
export default function ChatIntentModal({
  onSelect,
}: {
  onSelect: (intent: ChatIntent) => void;
}) {
  const options: {
    intent: ChatIntent;
    icon: typeof Wrench;
    title: string;
    blurb: string;
    accent: string;
  }[] = [
    {
      intent: "troubleshoot",
      icon: Wrench,
      title: "Fix a problem",
      blurb:
        "Something is faulty, noisy, leaking or stopped. You get a diagnosis, then a guided repair that starts with isolation and lockout/tagout.",
      accent: "amber",
    },
    {
      intent: "learn",
      icon: GraduationCap,
      title: "Learn how it works",
      blurb:
        "No fault to chase. You get an explanation of the machine, its components and how they interact, with the diagrams that show it.",
      accent: "sky",
    },
  ];

  return (
    <div className="flex flex-1 items-center justify-center p-6">
      <div className="w-full max-w-2xl">
        <h2 className="mb-1 text-center text-lg font-bold text-white">
          What do you need right now?
        </h2>
        <p className="mb-6 text-center text-sm text-slate-400">
          This sets how answers are written for the rest of this conversation.
        </p>

        <div className="grid gap-4 sm:grid-cols-2">
          {options.map(({ intent, icon: Icon, title, blurb, accent }) => (
            <button
              key={intent}
              onClick={() => onSelect(intent)}
              className={`group rounded-2xl border p-5 text-left transition ${
                accent === "amber"
                  ? "border-amber-500/30 bg-amber-500/5 hover:border-amber-500/60 hover:bg-amber-500/10"
                  : "border-sky-500/30 bg-sky-500/5 hover:border-sky-500/60 hover:bg-sky-500/10"
              }`}
            >
              <Icon
                size={22}
                className={accent === "amber" ? "text-amber-400" : "text-sky-400"}
              />
              <h3 className="mt-3 text-sm font-bold text-white">{title}</h3>
              <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{blurb}</p>
            </button>
          ))}
        </div>

        <p className="mt-5 text-center text-[11px] text-slate-500">
          You can start a new chat any time to switch.
        </p>
      </div>
    </div>
  );
}

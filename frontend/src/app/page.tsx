"use client";

import { useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { AppDispatch, RootState } from "../store/store";
import {
  askAssistant,
  fetchSessions,
  resolveSession,
  AskMode,
} from "../store/slices/assistantSlice";
import MachineSelector from "../components/MachineSelector";
import SessionSidebar from "../components/SessionSidebar";
import AnswerBubble from "../components/AnswerBubble";
import StepCard from "../components/StepCard";
import { Send, Loader2, Flame, ClipboardCheck, FileDown } from "lucide-react";

const EXAMPLES = [
  "The machine is vibrating more than usual — what should I check?",
  "It keeps overheating during a long run. What causes that?",
  "Walk me through replacing the drive belt.",
];

export default function AskPage() {
  const dispatch = useDispatch<AppDispatch>();
  const { messages, isAsking, activeSessionId, selectedMachineId, contextSource, isResolving } =
    useSelector((s: RootState) => s.assistant);

  const [input, setInput] = useState("");
  const [fixText, setFixText] = useState("");
  const [showResolve, setShowResolve] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, isAsking]);

  const send = (query: string, mode: AskMode = "answer") => {
    const text = query.trim();
    if (!text || isAsking) return;
    setInput("");
    dispatch(askAssistant({ query: text, mode })).then(() => dispatch(fetchSessions()));
  };

  const lastAgentIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "agent") return i;
    }
    return -1;
  })();
  const inWizard = lastAgentIndex >= 0 && messages[lastAgentIndex].type === "wizard_step";

  const submitFix = () => {
    if (!activeSessionId || !fixText.trim()) return;
    dispatch(resolveSession({ sessionId: activeSessionId, operatorFix: fixText.trim() })).then(
      () => {
        setFixText("");
        setShowResolve(false);
      }
    );
  };

  return (
    <div className="mx-auto flex h-[calc(100vh-65px)] max-w-7xl gap-6 p-4 md:p-6">
      <div className="hidden w-64 shrink-0 lg:block">
        <SessionSidebar />
      </div>

      <section className="flex min-w-0 flex-1 flex-col overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/40">
        <header className="flex flex-col gap-3 border-b border-slate-800 p-4 md:flex-row md:items-center md:justify-between">
          <div className="w-full md:max-w-sm">
            <MachineSelector />
          </div>
          <div className="flex items-center gap-2">
            {contextSource && (
              <span className="truncate rounded-full border border-slate-700 bg-slate-800/60 px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                {contextSource}
              </span>
            )}
            {activeSessionId && (
              <>
                <a
                  href={`/report/${activeSessionId}`}
                  className="flex items-center gap-1.5 rounded-xl border border-slate-700 px-3 py-2 text-xs font-bold text-slate-300 hover:bg-slate-800"
                >
                  <FileDown size={14} /> Report
                </a>
                <button
                  onClick={() => setShowResolve((v) => !v)}
                  className="flex items-center gap-1.5 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs font-bold text-emerald-400 hover:bg-emerald-500/20"
                >
                  <ClipboardCheck size={14} /> Record fix
                </button>
              </>
            )}
          </div>
        </header>

        {showResolve && activeSessionId && (
          <div className="border-b border-slate-800 bg-emerald-500/5 p-4">
            <label className="mb-2 block text-xs font-bold uppercase tracking-wider text-emerald-400">
              What actually fixed it?
            </label>
            <p className="mb-2 text-xs text-slate-400">
              Saved against this machine so the next person who asks sees it alongside the manual.
            </p>
            <div className="flex gap-2">
              <input
                value={fixText}
                onChange={(e) => setFixText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && submitFix()}
                placeholder="e.g. Replaced the worn drive belt and re-tensioned to 45 Nm"
                className="flex-1 rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-emerald-500"
              />
              <button
                onClick={submitFix}
                disabled={isResolving || !fixText.trim()}
                className="rounded-xl bg-emerald-600 px-4 py-2 text-sm font-bold text-white disabled:opacity-40"
              >
                {isResolving ? <Loader2 size={16} className="animate-spin" /> : "Save"}
              </button>
            </div>
          </div>
        )}

        <div className="flex-1 space-y-4 overflow-y-auto p-4 md:p-6">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="mb-4 rounded-2xl bg-orange-500/10 p-4">
                <Flame size={32} className="text-orange-500" />
              </div>
              <h2 className="text-xl font-black text-white">Ask about a machine problem</h2>
              <p className="mt-1 max-w-md text-sm text-slate-400">
                {selectedMachineId
                  ? "Answers are grounded in that machine's manual, with the relevant diagrams."
                  : "Select a machine above to get answers from its manual."}
              </p>
              <div className="mt-6 flex w-full max-w-lg flex-col gap-2">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => send(ex)}
                    className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-3 text-left text-sm text-slate-300 transition-colors hover:border-orange-500/40 hover:text-white"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className="space-y-2">
              <AnswerBubble message={m} />
              {i === lastAgentIndex && !isAsking && (
                <StepCard
                  visible
                  inWizard={inWizard}
                  disabled={isAsking}
                  onAction={(msg, mode) => send(msg, mode)}
                />
              )}
            </div>
          ))}

          {isAsking && (
            <div className="flex items-center gap-3 pl-11 text-sm text-slate-500">
              <Loader2 size={16} className="animate-spin text-orange-500" />
              Searching the manual…
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="border-t border-slate-800 p-4">
          <div className="flex items-end gap-2">
            <textarea
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(input);
                }
              }}
              placeholder="Describe the problem…"
              className="max-h-40 flex-1 resize-none rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none focus:border-orange-500"
            />
            <button
              onClick={() => send(input)}
              disabled={isAsking || !input.trim()}
              aria-label="Send"
              className="rounded-xl bg-orange-600 p-3 text-white transition-colors hover:bg-orange-500 disabled:opacity-40"
            >
              {isAsking ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

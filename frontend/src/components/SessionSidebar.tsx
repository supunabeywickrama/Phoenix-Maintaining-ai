"use client";

import { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { AppDispatch, RootState } from "../store/store";
import {
  fetchSessions,
  openSession,
  deleteSession,
  startNewSession,
} from "../store/slices/assistantSlice";
import { Plus, MessageSquare, Trash2, CheckCircle2 } from "lucide-react";

export default function SessionSidebar() {
  const dispatch = useDispatch<AppDispatch>();
  const { sessions, activeSessionId } = useSelector((s: RootState) => s.assistant);

  useEffect(() => {
    dispatch(fetchSessions());
  }, [dispatch]);

  return (
    <aside className="flex h-full flex-col gap-3">
      <button
        onClick={() => dispatch(startNewSession())}
        className="flex items-center justify-center gap-2 rounded-xl bg-orange-600 px-4 py-3 text-sm font-bold text-white transition-colors hover:bg-orange-500"
      >
        <Plus size={16} /> New enquiry
      </button>

      <div className="text-[10px] font-black uppercase tracking-[0.18em] text-slate-500">
        History
      </div>

      <div className="flex-1 space-y-1 overflow-y-auto">
        {sessions.length === 0 && (
          <p className="text-xs italic text-slate-600">No previous enquiries.</p>
        )}

        {sessions.map((s) => (
          <div
            key={s.id}
            className={`group flex items-center gap-2 rounded-xl border px-3 py-2 transition-colors ${
              activeSessionId === s.id
                ? "border-orange-500/30 bg-orange-500/10"
                : "border-transparent hover:bg-slate-800/60"
            }`}
          >
            <button
              onClick={() => dispatch(openSession(s.id))}
              className="flex min-w-0 flex-1 items-center gap-2 text-left"
            >
              {s.resolved ? (
                <CheckCircle2 size={14} className="shrink-0 text-emerald-500" />
              ) : (
                <MessageSquare size={14} className="shrink-0 text-slate-500" />
              )}
              <span className="min-w-0">
                <span className="block truncate text-xs font-bold text-slate-200">
                  {s.title}
                </span>
                <span className="block truncate text-[10px] text-slate-500">
                  {s.machine_id || "No machine"} · {s.timestamp}
                </span>
              </span>
            </button>

            <button
              onClick={() => dispatch(deleteSession(s.id))}
              aria-label={`Delete ${s.title}`}
              className="shrink-0 rounded-lg p-1 text-slate-600 opacity-0 transition-all hover:bg-red-500/10 hover:text-red-400 group-hover:opacity-100"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}

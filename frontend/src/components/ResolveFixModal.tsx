"use client";

import { useEffect, useRef, useState } from "react";
import { useDispatch } from "react-redux";
import { AppDispatch } from "../store/store";
import {
  fetchResolveDraft,
  resolveSession,
  FixMethod,
} from "../store/slices/assistantSlice";
import { CheckCircle2, Hand, Bot, Wrench, Loader2, Sparkles, X } from "lucide-react";

const ENGINEER_KEY = "phoenix.engineerName";

const METHODS: { value: FixMethod; label: string; hint: string; icon: typeof Hand }[] = [
  { value: "hands_on", label: "By hand", hint: "My own experience", icon: Hand },
  { value: "system_guided", label: "System instructions", hint: "Followed the steps given here", icon: Bot },
  { value: "both", label: "Both", hint: "Some of each", icon: Wrench },
];

function readEngineer(): string {
  try {
    return localStorage.getItem(ENGINEER_KEY) || "";
  } catch {
    return "";
  }
}

function saveEngineer(name: string) {
  try {
    localStorage.setItem(ENGINEER_KEY, name);
  } catch {
    /* storage unavailable — the name just won't be remembered */
  }
}

interface ResolveFixModalProps {
  sessionId: number;
  machineLabel: string;
  onClose: () => void;
}

type Errors = Partial<Record<"engineer" | "root_cause" | "actions" | "method", string>>;

/**
 * The engineer's record of how the machine was restored. Saved to the session
 * and to fix memory, so the next report of this fault on this machine opens
 * with it. Draft fields come from the conversation and are marked as such; the
 * engineer's name and method are always theirs to give.
 */
export default function ResolveFixModal({ sessionId, machineLabel, onClose }: ResolveFixModalProps) {
  const dispatch = useDispatch<AppDispatch>();
  // Lazy initial value: the dialog only ever mounts after a click, so reading
  // localStorage here is safe and avoids a second render to fill the name in.
  const [engineer, setEngineer] = useState(readEngineer);
  const [rootCause, setRootCause] = useState("");
  const [actions, setActions] = useState("");
  const [parts, setParts] = useState("");
  const [method, setMethod] = useState<FixMethod | null>(null);
  const [drafting, setDrafting] = useState(true);
  const [drafted, setDrafted] = useState(false);
  const [errors, setErrors] = useState<Errors>({});
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const firstFieldRef = useRef<HTMLInputElement>(null);

  // Draft from the chat, fetched once on open. Fields the engineer has already
  // typed into are never overwritten by a late draft.
  useEffect(() => {
    firstFieldRef.current?.focus();
    let cancelled = false;
    dispatch(fetchResolveDraft(sessionId))
      .unwrap()
      .then((draft) => {
        if (cancelled) return;
        setRootCause((v) => v || draft.root_cause || "");
        setActions((v) => v || draft.actions || "");
        setParts((v) => v || draft.parts_replaced || "");
        setDrafted(Boolean(draft.root_cause || draft.actions || draft.parts_replaced));
      })
      .catch(() => {
        /* no draft — the form simply starts empty */
      })
      .finally(() => !cancelled && setDrafting(false));
    return () => {
      cancelled = true;
    };
  }, [dispatch, sessionId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !saving && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, saving]);

  useEffect(() => {
    if (!saved) return;
    const t = setTimeout(onClose, 2200);
    return () => clearTimeout(t);
  }, [saved, onClose]);

  const submit = () => {
    const next: Errors = {};
    if (!engineer.trim()) next.engineer = "Enter the engineer's name.";
    if (!rootCause.trim()) next.root_cause = "Say what was actually wrong.";
    if (!actions.trim()) next.actions = "Describe what was done to restore the machine.";
    if (!method) next.method = "Choose how the problem was solved.";
    setErrors(next);
    if (Object.keys(next).length || !method) return;

    setSaving(true);
    setSaveError(null);
    saveEngineer(engineer.trim());
    dispatch(
      resolveSession({
        sessionId,
        fix: {
          engineer: engineer.trim(),
          root_cause: rootCause.trim(),
          actions: actions.trim(),
          method,
          parts_replaced: parts.trim() || undefined,
        },
      })
    )
      .unwrap()
      .then(() => setSaved(true))
      .catch((e: { message?: string }) =>
        setSaveError(e?.message || "The fix could not be saved. Check the connection and try again.")
      )
      .finally(() => setSaving(false));
  };

  const fieldClass = (bad?: string) =>
    `w-full rounded-xl border bg-slate-950 px-3 py-2 text-sm text-white outline-none transition-colors focus:border-emerald-500 ${
      bad ? "border-red-500/60" : "border-slate-700"
    }`;

  // A plain element, not a component defined here: a component declared inside
  // render is a new type every keystroke, which would remount the textarea and
  // drop focus while typing.
  const shimmer = (
    <div className="draft-shimmer h-[4.5rem] rounded-xl motion-safe:animate-shimmer" aria-hidden />
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-sm"
      onMouseDown={(e) => e.target === e.currentTarget && !saving && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="resolve-title"
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl motion-safe:animate-dialog-in"
      >
        {saved ? (
          <div className="flex flex-col items-center px-6 py-10 text-center">
            <div className="mb-4 rounded-full bg-emerald-500/15 p-4 motion-safe:animate-solved-pop">
              <CheckCircle2 size={40} className="text-emerald-400" />
            </div>
            <h2 className="text-lg font-black text-white">Fix saved</h2>
            <p className="mt-1 max-w-sm text-sm text-slate-400">
              The next time this fault is reported on {machineLabel}, this fix is shown first.
            </p>
          </div>
        ) : (
          <>
            <header className="flex items-start justify-between gap-3 border-b border-slate-800 p-5">
              <div>
                <h2 id="resolve-title" className="text-base font-black text-white">
                  How was the machine restored?
                </h2>
                <p className="mt-0.5 text-xs text-slate-400">
                  Saved against {machineLabel} so the next engineer sees what worked.
                </p>
              </div>
              <button
                onClick={onClose}
                disabled={saving}
                aria-label="Close"
                className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-800 hover:text-white"
              >
                <X size={18} />
              </button>
            </header>

            <div className="space-y-4 p-5">
              {!drafting && drafted && (
                <p className="flex items-center gap-1.5 rounded-lg border border-sky-500/25 bg-sky-500/5 px-3 py-2 text-[11px] text-sky-300">
                  <Sparkles size={12} /> Drafted from the chat — check and correct it before saving.
                </p>
              )}

              <div>
                <label htmlFor="fix-engineer" className="mb-1 block text-xs font-bold text-slate-300">
                  Engineer name <span className="text-red-400">*</span>
                </label>
                <input
                  id="fix-engineer"
                  ref={firstFieldRef}
                  value={engineer}
                  onChange={(e) => setEngineer(e.target.value)}
                  placeholder="e.g. J. Perera"
                  className={fieldClass(errors.engineer)}
                />
                {errors.engineer && <p className="mt-1 text-[11px] text-red-400">{errors.engineer}</p>}
              </div>

              <div>
                <label htmlFor="fix-cause" className="mb-1 block text-xs font-bold text-slate-300">
                  What was actually wrong? <span className="text-red-400">*</span>
                </label>
                {drafting ? shimmer : (
                  <textarea
                    id="fix-cause"
                    rows={2}
                    value={rootCause}
                    onChange={(e) => setRootCause(e.target.value)}
                    placeholder="e.g. Coolant pump impeller worn, flow too low to cool the injection module"
                    className={fieldClass(errors.root_cause)}
                  />
                )}
                {errors.root_cause && <p className="mt-1 text-[11px] text-red-400">{errors.root_cause}</p>}
              </div>

              <div>
                <label htmlFor="fix-actions" className="mb-1 block text-xs font-bold text-slate-300">
                  What did you do to restore the system? <span className="text-red-400">*</span>
                </label>
                {drafting ? shimmer : (
                  <textarea
                    id="fix-actions"
                    rows={4}
                    value={actions}
                    onChange={(e) => setActions(e.target.value)}
                    placeholder={"1. Isolated and locked out the machine\n2. Replaced the coolant pump\n3. Bled the circuit and test-ran for 30 min"}
                    className={fieldClass(errors.actions)}
                  />
                )}
                {errors.actions && <p className="mt-1 text-[11px] text-red-400">{errors.actions}</p>}
              </div>

              <fieldset>
                <legend className="mb-1.5 block text-xs font-bold text-slate-300">
                  How was it solved? <span className="text-red-400">*</span>
                </legend>
                <div className="grid gap-2 sm:grid-cols-3">
                  {METHODS.map(({ value, label, hint, icon: Icon }) => {
                    const on = method === value;
                    return (
                      <label
                        key={value}
                        className={`flex cursor-pointer flex-col gap-1 rounded-xl border p-3 transition-colors focus-within:ring-2 focus-within:ring-emerald-500/50 ${
                          on
                            ? "border-emerald-500/60 bg-emerald-500/10"
                            : errors.method
                            ? "border-red-500/40 hover:border-slate-500"
                            : "border-slate-700 hover:border-slate-500"
                        }`}
                      >
                        <input
                          type="radio"
                          name="fix-method"
                          value={value}
                          checked={on}
                          onChange={() => setMethod(value)}
                          className="sr-only"
                        />
                        <span className={`flex items-center gap-1.5 text-xs font-bold ${on ? "text-emerald-300" : "text-white"}`}>
                          <Icon size={14} /> {label}
                        </span>
                        <span className="text-[11px] text-slate-500">{hint}</span>
                      </label>
                    );
                  })}
                </div>
                {errors.method && <p className="mt-1 text-[11px] text-red-400">{errors.method}</p>}
              </fieldset>

              <div>
                <label htmlFor="fix-parts" className="mb-1 block text-xs font-bold text-slate-300">
                  Parts replaced <span className="font-normal text-slate-500">(optional)</span>
                </label>
                <input
                  id="fix-parts"
                  value={parts}
                  onChange={(e) => setParts(e.target.value)}
                  placeholder="e.g. Coolant pump 750-CP02"
                  className={fieldClass()}
                />
              </div>

              {saveError && (
                <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                  {saveError}
                </p>
              )}
            </div>

            <footer className="flex justify-end gap-2 border-t border-slate-800 p-4">
              <button
                onClick={onClose}
                disabled={saving}
                className="rounded-xl border border-slate-700 px-4 py-2 text-sm font-bold text-slate-300 hover:bg-slate-800"
              >
                Cancel
              </button>
              <button
                onClick={submit}
                disabled={saving}
                className="flex items-center gap-2 rounded-xl bg-emerald-600 px-4 py-2 text-sm font-bold text-white hover:bg-emerald-500 disabled:opacity-50"
              >
                {saving ? <Loader2 size={16} className="animate-spin" /> : <CheckCircle2 size={16} />}
                Save fix
              </button>
            </footer>
          </>
        )}
      </div>
    </div>
  );
}

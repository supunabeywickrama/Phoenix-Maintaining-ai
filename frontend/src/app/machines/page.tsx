"use client";

import { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { AppDispatch, RootState } from "../../store/store";
import {
  fetchMachines,
  saveMachine,
  deleteMachine,
  clearStatus,
} from "../../store/slices/machinesSlice";
import { fetchManuals } from "../../store/slices/manualsSlice";
import {
  Cog,
  Plus,
  MapPin,
  Trash2,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  BookOpen,
} from "lucide-react";

const EMPTY = { machine_id: "", name: "", location: "", manual_id: "" };

export default function MachinesPage() {
  const dispatch = useDispatch<AppDispatch>();
  const { items, isSaving, status, error } = useSelector((s: RootState) => s.machines);
  const manuals = useSelector((s: RootState) => s.manuals.items);

  const [form, setForm] = useState(EMPTY);

  useEffect(() => {
    dispatch(fetchMachines());
    dispatch(fetchManuals());
  }, [dispatch]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    dispatch(clearStatus());
    dispatch(saveMachine(form)).then((res) => {
      if (res.meta.requestStatus === "fulfilled") setForm(EMPTY);
    });
  };

  const remove = (id: string) => {
    if (!confirm(`Remove ${id} from the registry?`)) return;
    dispatch(deleteMachine(id));
  };

  return (
    <div className="mx-auto max-w-6xl p-4 md:p-8">
      <header className="mb-8">
        <h1 className="flex items-center gap-3 text-2xl font-black text-white md:text-3xl">
          <Cog className="text-orange-500" size={30} />
          Machines
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Link each machine to its manual so questions are answered from the right document.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-3">
        <form
          onSubmit={submit}
          className="h-fit rounded-2xl border border-slate-800 bg-slate-900 p-6"
        >
          <h2 className="mb-5 flex items-center gap-2 text-lg font-bold text-white">
            <Plus size={18} className="text-orange-400" /> Add a machine
          </h2>

          <label className="mb-2 block text-[11px] font-bold uppercase tracking-widest text-slate-500">
            Machine ID
          </label>
          <input
            required
            value={form.machine_id}
            onChange={(e) =>
              setForm({ ...form, machine_id: e.target.value.toUpperCase().replace(/\s/g, "_") })
            }
            placeholder="e.g. PRESS-04"
            className="mb-4 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none focus:border-orange-500"
          />

          <label className="mb-2 block text-[11px] font-bold uppercase tracking-widest text-slate-500">
            Name
          </label>
          <input
            required
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="e.g. Hydraulic Press"
            className="mb-4 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none focus:border-orange-500"
          />

          <label className="mb-2 block text-[11px] font-bold uppercase tracking-widest text-slate-500">
            Location
          </label>
          <div className="relative mb-4">
            <MapPin className="absolute left-4 top-3.5 text-slate-500" size={15} />
            <input
              value={form.location}
              onChange={(e) => setForm({ ...form, location: e.target.value })}
              placeholder="e.g. Line 2"
              className="w-full rounded-xl border border-slate-700 bg-slate-950 py-3 pl-11 pr-4 text-sm text-white outline-none focus:border-orange-500"
            />
          </div>

          <label className="mb-2 block text-[11px] font-bold uppercase tracking-widest text-slate-500">
            Manual
          </label>
          <select
            required
            value={form.manual_id}
            onChange={(e) => setForm({ ...form, manual_id: e.target.value })}
            className="mb-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none focus:border-orange-500"
          >
            <option value="">Select a manual…</option>
            {manuals.map((m) => (
              <option key={m.manual_id} value={m.manual_id}>
                {m.manual_id} ({m.chunks} sections)
              </option>
            ))}
          </select>
          {manuals.length === 0 && (
            <p className="mb-4 flex items-center gap-1.5 text-[11px] text-amber-400">
              <AlertTriangle size={12} /> No manuals yet — upload one first.
            </p>
          )}

          <button
            type="submit"
            disabled={isSaving}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-orange-600 px-6 py-3.5 font-bold text-white transition-colors hover:bg-orange-500 disabled:bg-slate-800 disabled:text-slate-500"
          >
            {isSaving ? <Loader2 size={18} className="animate-spin" /> : <Plus size={18} />}
            Save machine
          </button>

          {status && !error && (
            <div className="mt-4 flex items-start gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-3 text-xs font-medium text-emerald-400">
              <CheckCircle2 size={15} className="mt-0.5 shrink-0" /> {status}
            </div>
          )}
          {error && (
            <div className="mt-4 flex items-start gap-2 rounded-xl border border-red-500/20 bg-red-500/10 p-3 text-xs font-medium text-red-400">
              <AlertTriangle size={15} className="mt-0.5 shrink-0" /> {error}
            </div>
          )}
        </form>

        <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 lg:col-span-2">
          <div className="flex items-center justify-between border-b border-slate-800 p-5">
            <h2 className="text-lg font-bold text-white">Registry</h2>
            <span className="rounded-full bg-slate-800 px-3 py-1 text-[10px] font-black uppercase tracking-wider text-slate-400">
              {items.length} machine{items.length === 1 ? "" : "s"}
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-sm">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-950/40">
                  <th className="px-5 py-3 text-[10px] font-black uppercase tracking-widest text-slate-500">Machine</th>
                  <th className="px-5 py-3 text-[10px] font-black uppercase tracking-widest text-slate-500">Location</th>
                  <th className="px-5 py-3 text-[10px] font-black uppercase tracking-widest text-slate-500">Manual</th>
                  <th className="px-5 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {items.map((m) => (
                  <tr key={m.machine_id} className="hover:bg-slate-800/20">
                    <td className="px-5 py-4">
                      <div className="font-bold text-white">{m.name}</div>
                      <div className="font-mono text-[11px] text-orange-400">{m.machine_id}</div>
                    </td>
                    <td className="px-5 py-4 text-xs text-slate-400">{m.location || "—"}</td>
                    <td className="px-5 py-4">
                      <div className="flex items-center gap-1.5 text-xs text-slate-300">
                        <BookOpen size={12} className="text-slate-500" />
                        {m.manual_id}
                      </div>
                      {m.manual_chunks === 0 && (
                        <div className="mt-1 flex items-center gap-1 text-[10px] font-bold text-amber-400">
                          <AlertTriangle size={10} /> not ingested
                        </div>
                      )}
                    </td>
                    <td className="px-5 py-4 text-right">
                      <button
                        onClick={() => remove(m.machine_id)}
                        aria-label={`Remove ${m.machine_id}`}
                        className="rounded-xl p-2 text-slate-500 transition-colors hover:bg-red-500/10 hover:text-red-400"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
                {items.length === 0 && (
                  <tr>
                    <td colSpan={4} className="px-5 py-12 text-center text-sm italic text-slate-600">
                      No machines registered yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

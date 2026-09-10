"use client";

import { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { AppDispatch, RootState } from "../../store/store";
import { fetchManuals, uploadManual, clearStatus } from "../../store/slices/manualsSlice";
import {
  BookOpen,
  UploadCloud,
  FileText,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
} from "lucide-react";

export default function ManualsPage() {
  const dispatch = useDispatch<AppDispatch>();
  const { items, isUploading, status, error } = useSelector((s: RootState) => s.manuals);

  const [manualId, setManualId] = useState("");
  const [file, setFile] = useState<File | null>(null);

  useEffect(() => {
    dispatch(fetchManuals());
  }, [dispatch]);

  const submit = () => {
    if (!manualId.trim() || !file) return;
    dispatch(clearStatus());
    dispatch(uploadManual({ manualId: manualId.trim(), file })).then((res) => {
      if (res.meta.requestStatus === "fulfilled") {
        setManualId("");
        setFile(null);
      }
    });
  };

  return (
    <div className="mx-auto max-w-6xl p-4 md:p-8">
      <header className="mb-8">
        <h1 className="flex items-center gap-3 text-2xl font-black text-white md:text-3xl">
          <BookOpen className="text-orange-500" size={30} />
          Manuals
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload equipment documentation. Text, tables and diagrams are all made searchable.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-5">
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 lg:col-span-2">
          <h2 className="mb-5 flex items-center gap-2 text-lg font-bold text-white">
            <UploadCloud size={18} className="text-orange-400" />
            Upload a manual
          </h2>

          <label className="mb-2 block text-[11px] font-bold uppercase tracking-widest text-slate-500">
            Manual identifier
          </label>
          <input
            value={manualId}
            onChange={(e) => setManualId(e.target.value.replace(/[^a-zA-Z0-9_-]/g, ""))}
            placeholder="e.g. PRESS_4000_V2"
            disabled={isUploading}
            className="mb-1 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white outline-none focus:border-orange-500 disabled:opacity-50"
          />
          <p className="mb-5 text-[11px] text-slate-500">
            Letters, numbers, hyphens and underscores. Machines link to this id.
          </p>

          <label className="mb-2 block text-[11px] font-bold uppercase tracking-widest text-slate-500">
            PDF document
          </label>
          <label
            className={`mb-5 flex cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed p-8 transition-colors ${
              file ? "border-orange-500/50 bg-orange-500/5" : "border-slate-700 hover:border-slate-600"
            } ${isUploading ? "pointer-events-none opacity-50" : ""}`}
          >
            <div className={`rounded-full p-3 ${file ? "bg-orange-500/20 text-orange-400" : "bg-slate-800 text-slate-500"}`}>
              <FileText size={26} />
            </div>
            <div className="text-center">
              <span className="block text-sm font-bold text-slate-200">
                {file ? file.name : "Choose a PDF"}
              </span>
              <span className="mt-0.5 block text-[11px] text-slate-500">
                {file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "Scanned manuals are fine"}
              </span>
            </div>
            <input
              type="file"
              accept="application/pdf"
              className="hidden"
              disabled={isUploading}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>

          <button
            onClick={submit}
            disabled={isUploading || !file || !manualId.trim()}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-orange-600 px-6 py-3.5 font-bold text-white transition-colors hover:bg-orange-500 disabled:bg-slate-800 disabled:text-slate-500"
          >
            {isUploading ? <Loader2 size={18} className="animate-spin" /> : <UploadCloud size={18} />}
            {isUploading ? "Processing…" : "Start ingestion"}
          </button>

          {isUploading && (
            <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
              Every diagram is analysed individually, so a long illustrated manual can take a
              while. You can leave this page open — progress continues on the server.
            </p>
          )}

          {status && !error && (
            <div className="mt-4 flex items-start gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-3 text-xs font-medium text-emerald-400">
              {isUploading ? (
                <Loader2 size={15} className="mt-0.5 shrink-0 animate-spin" />
              ) : (
                <CheckCircle2 size={15} className="mt-0.5 shrink-0" />
              )}
              {status}
            </div>
          )}
          {error && (
            <div className="mt-4 flex items-start gap-2 rounded-xl border border-red-500/20 bg-red-500/10 p-3 text-xs font-medium text-red-400">
              <AlertTriangle size={15} className="mt-0.5 shrink-0" />
              {error}
            </div>
          )}
        </div>

        <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 lg:col-span-3">
          <div className="flex items-center justify-between border-b border-slate-800 p-5">
            <h2 className="text-lg font-bold text-white">Knowledge base</h2>
            <span className="rounded-full bg-slate-800 px-3 py-1 text-[10px] font-black uppercase tracking-wider text-slate-400">
              {items.length} manual{items.length === 1 ? "" : "s"}
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-950/40">
                  <th className="px-5 py-3 text-[10px] font-black uppercase tracking-widest text-slate-500">Manual</th>
                  <th className="px-5 py-3 text-[10px] font-black uppercase tracking-widest text-slate-500">File</th>
                  <th className="px-5 py-3 text-right text-[10px] font-black uppercase tracking-widest text-slate-500">Sections</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {items.map((m) => (
                  <tr key={m.manual_id} className="hover:bg-slate-800/20">
                    <td className="px-5 py-4">
                      <span className="rounded-lg border border-orange-400/20 bg-orange-400/5 px-2 py-1 font-mono text-xs font-bold text-orange-400">
                        {m.manual_id}
                      </span>
                    </td>
                    <td className="px-5 py-4">
                      {m.url ? (
                        <a
                          href={m.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1.5 text-xs text-slate-300 hover:text-orange-400"
                        >
                          {m.filename || "Source PDF"} <ExternalLink size={11} />
                        </a>
                      ) : (
                        <span className="text-xs text-slate-500">{m.filename || "—"}</span>
                      )}
                    </td>
                    <td className="px-5 py-4 text-right">
                      <span className={`text-xs font-bold ${m.chunks > 0 ? "text-emerald-400" : "text-amber-400"}`}>
                        {m.chunks > 0 ? m.chunks : "empty"}
                      </span>
                    </td>
                  </tr>
                ))}
                {items.length === 0 && (
                  <tr>
                    <td colSpan={3} className="px-5 py-12 text-center text-sm italic text-slate-600">
                      No manuals yet. Upload one to get started.
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

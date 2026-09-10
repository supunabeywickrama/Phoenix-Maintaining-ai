"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { useDispatch, useSelector } from "react-redux";
import { AppDispatch, RootState } from "../../../store/store";
import { fetchReport } from "../../../store/slices/assistantSlice";
import { FileDown, Loader2, ArrowLeft, CheckCircle2 } from "lucide-react";
import Link from "next/link";

export default function ReportPage() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = Number(params?.sessionId);
  const dispatch = useDispatch<AppDispatch>();
  const report = useSelector((s: RootState) => s.assistant.report);
  const [building, setBuilding] = useState(false);

  useEffect(() => {
    if (Number.isFinite(sessionId)) dispatch(fetchReport(sessionId));
  }, [dispatch, sessionId]);

  /** Fetch an image and convert it for embedding; skipped if the host blocks CORS. */
  const toDataUrl = async (url: string): Promise<string | null> => {
    try {
      const res = await fetch(url, { mode: "cors" });
      if (!res.ok) return null;
      const blob = await res.blob();
      return await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(blob);
      });
    } catch {
      return null;
    }
  };

  const download = async () => {
    if (!report) return;
    setBuilding(true);
    try {
      const { default: jsPDF } = await import("jspdf");
      const doc = new jsPDF({ unit: "pt", format: "a4" });
      const margin = 48;
      const width = doc.internal.pageSize.getWidth() - margin * 2;
      let y = margin;

      const newPageIfNeeded = (needed: number) => {
        if (y + needed > doc.internal.pageSize.getHeight() - margin) {
          doc.addPage();
          y = margin;
        }
      };

      doc.setFontSize(20).setFont("helvetica", "bold");
      doc.text("Phoenix Industries — Maintenance Report", margin, y);
      y += 26;

      doc.setFontSize(10).setFont("helvetica", "normal").setTextColor(110);
      doc.text(
        `Machine: ${report.machineId || "—"}   |   Session #${report.sessionId}   |   ${report.timestamp}`,
        margin,
        y
      );
      y += 24;
      doc.setTextColor(0);

      const section = (title: string, body: string) => {
        newPageIfNeeded(60);
        doc.setFontSize(13).setFont("helvetica", "bold");
        doc.text(title, margin, y);
        y += 16;
        doc.setFontSize(11).setFont("helvetica", "normal");
        const lines = doc.splitTextToSize(body || "—", width);
        newPageIfNeeded(lines.length * 14);
        doc.text(lines, margin, y);
        y += lines.length * 14 + 14;
      };

      section("Problem", report.problemDescription);
      section("Diagnosis", report.diagnosis);

      newPageIfNeeded(40);
      doc.setFontSize(13).setFont("helvetica", "bold");
      doc.text("Solution steps", margin, y);
      y += 16;
      doc.setFontSize(11).setFont("helvetica", "normal");
      report.solutionSteps.forEach((step, i) => {
        const lines = doc.splitTextToSize(`${i + 1}. ${step}`, width - 12);
        newPageIfNeeded(lines.length * 14 + 6);
        doc.text(lines, margin + 6, y);
        y += lines.length * 14 + 6;
      });

      for (const img of report.images.slice(0, 6)) {
        const data = await toDataUrl(img.url);
        if (!data) continue;
        newPageIfNeeded(220);
        try {
          doc.addImage(data, "PNG", margin, y, 220, 160, undefined, "FAST");
          y += 168;
          doc.setFontSize(9).setTextColor(110);
          doc.text(img.caption, margin, y);
          doc.setTextColor(0);
          y += 18;
        } catch {
          /* unsupported image format — omit rather than fail the export */
        }
      }

      doc.save(`phoenix-report-${report.sessionId}.pdf`);
    } finally {
      setBuilding(false);
    }
  };

  if (!report) {
    return (
      <div className="flex h-[60vh] items-center justify-center gap-3 text-slate-500">
        <Loader2 className="animate-spin text-orange-500" size={20} /> Building report…
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl p-4 md:p-8">
      <div className="mb-6 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 text-sm font-bold text-slate-400 hover:text-white">
          <ArrowLeft size={16} /> Back
        </Link>
        <button
          onClick={download}
          disabled={building}
          className="flex items-center gap-2 rounded-xl bg-orange-600 px-4 py-2.5 text-sm font-bold text-white hover:bg-orange-500 disabled:opacity-50"
        >
          {building ? <Loader2 size={16} className="animate-spin" /> : <FileDown size={16} />}
          Download PDF
        </button>
      </div>

      <article className="rounded-2xl border border-slate-800 bg-slate-900 p-6 md:p-8">
        <h1 className="text-2xl font-black text-white">{report.title}</h1>
        <p className="mt-1 text-xs text-slate-500">
          {report.machineId || "No machine"} · Session #{report.sessionId} · {report.timestamp}
        </p>
        {report.resolvedAt && (
          <p className="mt-2 inline-flex items-center gap-1.5 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-3 py-1 text-[11px] font-bold text-emerald-400">
            <CheckCircle2 size={12} /> Resolved {report.resolvedAt}
          </p>
        )}

        <section className="mt-6">
          <h2 className="text-[11px] font-black uppercase tracking-widest text-orange-400">Problem</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-300">{report.problemDescription}</p>
        </section>

        <section className="mt-6">
          <h2 className="text-[11px] font-black uppercase tracking-widest text-orange-400">Diagnosis</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-300">{report.diagnosis}</p>
        </section>

        <section className="mt-6">
          <h2 className="text-[11px] font-black uppercase tracking-widest text-orange-400">Solution steps</h2>
          <ol className="mt-2 space-y-2">
            {report.solutionSteps.map((s, i) => (
              <li key={i} className="flex gap-3 text-sm text-slate-300">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-slate-800 text-[10px] font-bold text-white">
                  {i + 1}
                </span>
                {s}
              </li>
            ))}
            {report.solutionSteps.length === 0 && (
              <li className="text-sm italic text-slate-600">No steps captured.</li>
            )}
          </ol>
        </section>

        {report.images.length > 0 && (
          <section className="mt-6">
            <h2 className="text-[11px] font-black uppercase tracking-widest text-orange-400">Referenced figures</h2>
            <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-3">
              {report.images.map((img) => (
                <figure key={img.url}>
                  <img
                    src={img.url}
                    alt={img.caption}
                    className="w-full rounded-lg border border-slate-700 bg-white/5"
                  />
                  <figcaption className="mt-1 text-[10px] text-slate-500">{img.caption}</figcaption>
                </figure>
              ))}
            </div>
          </section>
        )}
      </article>
    </div>
  );
}

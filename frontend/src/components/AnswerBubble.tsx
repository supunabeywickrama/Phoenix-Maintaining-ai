"use client";

import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, User, AlertTriangle, X } from "lucide-react";
import type { ChatMessage } from "../store/slices/assistantSlice";

/** `[IMAGE_0]`, `[IMAGE_12]` … emitted inline by the retrieval prompts. */
const IMAGE_TAG = /\[IMAGE_(\d+)\]/g;
/** The wizard prefixes its replies with `[PHASE: Safety]`. */
const PHASE_TAG = /^\s*\[PHASE:\s*([^\]]+)\]\s*/i;
/** Summary mode ends with a tag used to offer the step-by-step follow-up. */
const SUGGESTION_TAG = /\[SUGGESTION:[^\]]*\]/gi;

type Segment =
  | { kind: "text"; value: string }
  | { kind: "image"; url: string; index: number };

/**
 * Split an answer into text and image segments so diagrams appear at the exact
 * step they illustrate. Without this the markers would render as literal
 * "[IMAGE_0]" text and the figures would be lost.
 */
function segment(content: string, images: string[]): Segment[] {
  const out: Segment[] = [];
  const used = new Set<number>();
  let cursor = 0;

  for (const match of content.matchAll(IMAGE_TAG)) {
    const idx = Number(match[1]);
    const at = match.index ?? 0;

    const before = content.slice(cursor, at);
    if (before.trim()) out.push({ kind: "text", value: before });

    if (images[idx]) {
      out.push({ kind: "image", url: images[idx], index: idx });
      used.add(idx);
    }
    cursor = at + match[0].length;
  }

  const tail = content.slice(cursor);
  if (tail.trim()) out.push({ kind: "text", value: tail });
  if (out.length === 0) out.push({ kind: "text", value: content });

  // Retrieved figures the model never referenced still belong to the answer.
  images.forEach((url, idx) => {
    if (!used.has(idx)) out.push({ kind: "image", url, index: idx });
  });

  return out;
}

export default function AnswerBubble({ message }: { message: ChatMessage }) {
  const [zoomed, setZoomed] = useState<string | null>(null);

  const { phase, segments } = useMemo(() => {
    if (message.role === "user") {
      return { phase: null, segments: [{ kind: "text", value: message.content }] as Segment[] };
    }
    let body = message.content.replace(SUGGESTION_TAG, "").trim();
    const phaseMatch = body.match(PHASE_TAG);
    if (phaseMatch) body = body.replace(PHASE_TAG, "");
    return { phase: phaseMatch?.[1]?.trim() ?? null, segments: segment(body, message.images || []) };
  }, [message]);

  if (message.role === "user") {
    return (
      <div className="flex justify-end gap-3">
        <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-orange-600 px-4 py-3 text-sm font-medium text-white shadow-lg shadow-orange-900/20">
          {message.content}
        </div>
        <div className="mt-1 h-8 w-8 shrink-0 rounded-full bg-slate-700 p-1.5">
          <User size={20} className="text-slate-300" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <div
        className={`mt-1 h-8 w-8 shrink-0 rounded-full p-1.5 ${
          message.isError ? "bg-red-500/20" : "bg-orange-500/15"
        }`}
      >
        {message.isError ? (
          <AlertTriangle size={20} className="text-red-400" />
        ) : (
          <Bot size={20} className="text-orange-400" />
        )}
      </div>

      <div
        className={`max-w-[85%] rounded-2xl rounded-tl-sm border px-4 py-3 ${
          message.isError
            ? "border-red-500/25 bg-red-500/5 text-red-300"
            : "border-slate-800 bg-slate-900 text-slate-200"
        }`}
      >
        {phase && (
          <div className="mb-2 inline-block rounded-full border border-orange-500/30 bg-orange-500/10 px-2.5 py-0.5 text-[10px] font-black uppercase tracking-wider text-orange-400">
            {phase}
          </div>
        )}

        <div className="answer text-sm">
          {segments.map((seg, i) =>
            seg.kind === "text" ? (
              <ReactMarkdown key={i} remarkPlugins={[remarkGfm]}>
                {seg.value}
              </ReactMarkdown>
            ) : (
              <figure key={i} className="my-3">
                {/* Plain <img>: these are Cloudinary URLs of arbitrary dimensions,
                    and next/image adds no value for content we do not control. */}
                <img
                  src={seg.url}
                  alt={`Manual figure ${seg.index + 1}`}
                  onClick={() => setZoomed(seg.url)}
                  className="max-h-80 w-auto cursor-zoom-in rounded-xl border border-slate-700 bg-white/5"
                />
                <figcaption className="mt-1 text-[11px] text-slate-500">
                  Figure {seg.index + 1} — from the manual. Click to enlarge.
                </figcaption>
              </figure>
            )
          )}
        </div>
      </div>

      {zoomed && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-6"
          onClick={() => setZoomed(null)}
        >
          <button
            className="absolute right-6 top-6 rounded-full bg-slate-800 p-2 text-slate-300 hover:text-white"
            aria-label="Close"
            onClick={() => setZoomed(null)}
          >
            <X size={20} />
          </button>
          <img src={zoomed} alt="Manual figure enlarged" className="max-h-full max-w-full rounded-xl" />
        </div>
      )}
    </div>
  );
}

"use client";

import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bot,
  User,
  AlertTriangle,
  X,
  Table as TableIcon,
  GitBranch,
  BarChart3,
  Image as ImageIcon,
  Layers,
} from "lucide-react";
import type { Attachment, ChatMessage } from "../store/slices/assistantSlice";

/** `[IMAGE_0]`, `[TABLE_1]` … emitted inline by the retrieval prompts. */
const ASSET_TAG = /\[(IMAGE|TABLE)_(\d+)\]/g;
/** The wizard prefixes its replies with `[PHASE: Safety]`. */
const PHASE_TAG = /^\s*\[PHASE:\s*([^\]]+)\]\s*/i;
/** Summary mode ends with a tag used to offer the step-by-step follow-up. */
const SUGGESTION_TAG = /\[SUGGESTION:[^\]]*\]/gi;

type Segment =
  | { kind: "text"; value: string }
  | { kind: "asset"; asset: Attachment };

/** Icon and label per asset kind, so a reader can tell a wiring schematic from a
 *  torque table before opening it. */
const KIND_META: Record<string, { icon: typeof TableIcon; label: string }> = {
  table: { icon: TableIcon, label: "Table" },
  flowchart: { icon: GitBranch, label: "Flow chart" },
  chart: { icon: BarChart3, label: "Chart" },
  schematic: { icon: Layers, label: "Schematic" },
  exploded_view: { icon: Layers, label: "Exploded view" },
  photo: { icon: ImageIcon, label: "Photo" },
  diagram: { icon: ImageIcon, label: "Diagram" },
};

function kindMeta(kind?: string) {
  return KIND_META[(kind || "diagram").toLowerCase()] ?? KIND_META.diagram;
}

/**
 * Older messages (and the legacy API shape) only carry a flat list of image URLs.
 * Promote those to attachments so one rendering path covers both.
 */
function normalise(message: ChatMessage): Attachment[] {
  if (message.attachments?.length) return message.attachments;
  return (message.images || []).map((url, i) => ({
    tag: `IMAGE_${i}`,
    type: "image" as const,
    kind: "diagram",
    title: `Figure ${i + 1}`,
    page: null,
    url,
  }));
}

/**
 * Split an answer into text and asset segments so each figure or table appears at
 * the exact point it is being talked about. Without this the markers would render
 * as literal "[IMAGE_0]" text and the material would be lost.
 */
function segment(content: string, assets: Attachment[]): Segment[] {
  const byTag = new Map(assets.map((a) => [a.tag, a]));
  const out: Segment[] = [];
  const used = new Set<string>();
  let cursor = 0;

  for (const match of content.matchAll(ASSET_TAG)) {
    const tag = `${match[1]}_${match[2]}`;
    const at = match.index ?? 0;

    const before = content.slice(cursor, at);
    if (before.trim()) out.push({ kind: "text", value: before });

    // Render each asset once, however many times the model referenced it —
    // repeated tags would otherwise stack the same table down the page. Later
    // mentions just drop the marker, leaving the surrounding sentence intact.
    const asset = byTag.get(tag);
    if (asset && !used.has(tag)) {
      out.push({ kind: "asset", asset });
      used.add(tag);
    }
    cursor = at + match[0].length;
  }

  const tail = content.slice(cursor);
  if (tail.trim()) out.push({ kind: "text", value: tail });
  if (out.length === 0) out.push({ kind: "text", value: content });

  // Retrieved material the model never referenced still belongs to the answer,
  // appended in the order retrieval chose: full view, its parts, then tables.
  assets.forEach((a) => {
    if (!used.has(a.tag)) out.push({ kind: "asset", asset: a });
  });

  return out;
}

function AssetBlock({
  asset,
  onZoom,
}: {
  asset: Attachment;
  onZoom: (url: string) => void;
}) {
  const { icon: Icon, label } = kindMeta(asset.kind);
  const isPart = asset.role === "part";
  const caption = [
    label,
    isPart ? "· component" : null,
    asset.page != null ? `· page ${asset.page}` : null,
  ]
    .filter(Boolean)
    .join(" ");

  if (asset.type === "table" && asset.markdown) {
    return (
      <figure className="my-3 overflow-hidden rounded-xl border border-slate-700 bg-slate-950/60">
        <figcaption className="flex items-center gap-1.5 border-b border-slate-800 bg-slate-900/60 px-3 py-2 text-[11px] font-bold text-slate-300">
          <Icon size={13} className="text-sky-400" />
          {asset.title}
          <span className="font-normal text-slate-500">{caption}</span>
        </figcaption>
        {/* Tables from a manual can be wide; scroll rather than squash them. */}
        <div className="answer-table max-w-full overflow-x-auto p-2 text-xs">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{asset.markdown}</ReactMarkdown>
        </div>
      </figure>
    );
  }

  if (!asset.url) return null;

  return (
    <figure className="my-3">
      {/* Plain <img>: these are Cloudinary/local URLs of arbitrary dimensions,
          and next/image adds no value for content we do not control. */}
      <img
        src={asset.url}
        alt={asset.title}
        onClick={() => onZoom(asset.url as string)}
        className={`w-auto cursor-zoom-in rounded-xl border bg-white/5 ${
          isPart ? "max-h-56 border-slate-700/70" : "max-h-80 border-slate-700"
        }`}
      />
      <figcaption className="mt-1 flex items-center gap-1.5 text-[11px] text-slate-500">
        <Icon size={12} className={isPart ? "text-slate-500" : "text-orange-400"} />
        <span className="font-semibold text-slate-400">{asset.title}</span>
        <span>{caption} · click to enlarge</span>
      </figcaption>
    </figure>
  );
}

export default function AnswerBubble({ message }: { message: ChatMessage }) {
  const [zoomed, setZoomed] = useState<string | null>(null);

  // Shared between the user- and agent-message returns below (they return
  // separate element trees), so a click-to-zoom on either kind of image works.
  const zoomOverlay = zoomed && (
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
      <img src={zoomed} alt="Enlarged" className="max-h-full max-w-full rounded-xl" />
    </div>
  );

  const { phase, segments } = useMemo(() => {
    if (message.role === "user") {
      return { phase: null, segments: [{ kind: "text", value: message.content }] as Segment[] };
    }
    let body = message.content.replace(SUGGESTION_TAG, "").trim();
    const phaseMatch = body.match(PHASE_TAG);
    if (phaseMatch) body = body.replace(PHASE_TAG, "");
    return {
      phase: phaseMatch?.[1]?.trim() ?? null,
      segments: segment(body, normalise(message)),
    };
  }, [message]);

  if (message.role === "user") {
    return (
      <>
        <div className="flex justify-end gap-3">
          <div className="flex max-w-[85%] flex-col items-end gap-2">
            {/* A photo attached alongside the question — the user's own upload,
                not one of the manual's figures, so no caption/kind chrome here. */}
            {message.images?.[0] && (
              <img
                src={message.images[0]}
                alt="Attached to your message"
                onClick={() => setZoomed(message.images![0])}
                className="max-h-56 w-auto cursor-zoom-in rounded-2xl border border-orange-400/30"
              />
            )}
            <div className="rounded-2xl rounded-tr-sm bg-orange-600 px-4 py-3 text-sm font-medium text-white shadow-lg shadow-orange-900/20">
              {message.content}
            </div>
          </div>
          <div className="mt-1 h-8 w-8 shrink-0 rounded-full bg-slate-700 p-1.5">
            <User size={20} className="text-slate-300" />
          </div>
        </div>
        {zoomOverlay}
      </>
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
              <AssetBlock key={i} asset={seg.asset} onZoom={setZoomed} />
            )
          )}
        </div>
      </div>

      {zoomOverlay}
    </div>
  );
}

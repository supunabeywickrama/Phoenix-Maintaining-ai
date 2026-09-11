import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { API_BASE, apiFetch } from "../api";

export interface ManualSummary {
  manual_id: string;
  filename: string | null;
  url: string | null;
  created_at: string | null;
  chunks: number;
}

interface ManualsState {
  items: ManualSummary[];
  isUploading: boolean;
  status: string | null;
  error: string | null;
}

const initialState: ManualsState = {
  items: [],
  isUploading: false,
  status: null,
  error: null,
};

export const fetchManuals = createAsyncThunk("manuals/fetch", async () =>
  apiFetch<ManualSummary[]>("/api/manuals")
);

interface IngestJob {
  manual_id: string;
  status: "processing" | "success" | "failed" | "unknown";
  chunks?: number | null;
  error?: string | null;
}

const POLL_MS = 3000;
const MAX_WAIT_MS = 60 * 60 * 1000; // an illustrated manual can take ~an hour

/**
 * Upload a manual and wait for the background pipeline to finish.
 *
 * The upload call only queues the job — a large manual makes one vision call per
 * figure, far longer than a browser will hold a request open — so completion is
 * observed by polling instead.
 */
export const uploadManual = createAsyncThunk(
  "manuals/upload",
  async ({ manualId, file }: { manualId: string; file: File }, { dispatch }) => {
    const form = new FormData();
    form.append("manual_id", manualId);
    form.append("file", file);

    await apiFetch<{ manual_id: string }>("/ingest-manual", {
      method: "POST",
      body: form,
    });

    const startedAt = Date.now();
    let unknownStreak = 0;

    while (Date.now() - startedAt < MAX_WAIT_MS) {
      await new Promise((r) => setTimeout(r, POLL_MS));

      const res = await fetch(
        `${API_BASE}/ingest-manual/status/${encodeURIComponent(manualId)}`
      );
      if (!res.ok) continue;
      const job: IngestJob = await res.json();

      if (job.status === "success") {
        dispatch(fetchManuals());
        return { manualId, chunks: job.chunks ?? 0 };
      }
      if (job.status === "failed") {
        throw new Error(job.error || "Ingestion failed");
      }
      // 'unknown' means the backend restarted and lost its job registry.
      unknownStreak = job.status === "unknown" ? unknownStreak + 1 : 0;
      if (unknownStreak >= 3) {
        throw new Error("Lost track of the ingestion job — did the backend restart?");
      }
    }

    throw new Error("Ingestion is taking unusually long. Check the backend logs.");
  }
);

export const deleteManual = createAsyncThunk(
  "manuals/delete",
  async (manualId: string, { dispatch }) => {
    const res = await apiFetch<{
      manual_id: string;
      chunks_removed: number;
      orphaned_machines: string[];
    }>(`/api/manuals/${encodeURIComponent(manualId)}`, { method: "DELETE" });
    dispatch(fetchManuals());
    return res;
  }
);

export const renameManual = createAsyncThunk(
  "manuals/rename",
  async (
    { manualId, newManualId }: { manualId: string; newManualId: string },
    { dispatch }
  ) => {
    const res = await apiFetch<{ manual_id: string; machines_updated: number }>(
      `/api/manuals/${encodeURIComponent(manualId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_manual_id: newManualId }),
      }
    );
    dispatch(fetchManuals());
    return res;
  }
);

const manualsSlice = createSlice({
  name: "manuals",
  initialState,
  reducers: {
    clearStatus(state) {
      state.status = null;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchManuals.fulfilled, (state, action) => {
        state.items = action.payload;
      })
      .addCase(uploadManual.pending, (state) => {
        state.isUploading = true;
        state.error = null;
        state.status = "Uploading and vectorising — this can take a while for a large manual.";
      })
      .addCase(uploadManual.fulfilled, (state, action) => {
        state.isUploading = false;
        state.status = `Ingested "${action.payload.manualId}" — ${action.payload.chunks} searchable sections.`;
      })
      .addCase(uploadManual.rejected, (state, action) => {
        state.isUploading = false;
        state.error = action.error.message || "Upload failed";
        state.status = null;
      })
      .addCase(deleteManual.fulfilled, (state, action) => {
        const { manual_id, chunks_removed, orphaned_machines } = action.payload;
        const orphanNote = orphaned_machines.length
          ? ` ${orphaned_machines.length} machine(s) still point at it: ${orphaned_machines.join(", ")}.`
          : "";
        state.status = `Deleted "${manual_id}" — ${chunks_removed} sections removed.${orphanNote}`;
        state.error = null;
      })
      .addCase(deleteManual.rejected, (state, action) => {
        state.error = action.error.message || "Delete failed";
      })
      .addCase(renameManual.fulfilled, (state, action) => {
        state.status = `Renamed to "${action.payload.manual_id}" — ${action.payload.machines_updated} machine link(s) updated.`;
        state.error = null;
      })
      .addCase(renameManual.rejected, (state, action) => {
        state.error = action.error.message || "Rename failed";
      });
  },
});

export const { clearStatus } = manualsSlice.actions;
export default manualsSlice.reducer;

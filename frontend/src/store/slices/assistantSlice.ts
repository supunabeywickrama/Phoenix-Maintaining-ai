import { createSlice, createAsyncThunk, PayloadAction } from "@reduxjs/toolkit";
import { apiFetch } from "../api";

export type AskMode = "answer" | "wizard";
/** Why this chat was opened. Chosen once, when the conversation starts. */
export type ChatIntent = "troubleshoot" | "learn";

/**
 * Something shown alongside an answer. Not just pictures: a manual's knowledge
 * also lives in schematics, charts, flow diagrams and spec tables, and a table
 * is far more use rendered as a table than paraphrased into a sentence.
 */
export interface Attachment {
  /** Matches the inline marker in the answer text, e.g. "IMAGE_0" / "TABLE_1". */
  tag: string;
  type: "image" | "table";
  kind: string;
  role?: "full" | "part" | null;
  title: string;
  page: number | null;
  url?: string;
  markdown?: string;
}

/** How an engineer restored the machine. */
export type FixMethod = "hands_on" | "system_guided" | "both";

/** A confirmed fix from earlier on the same machine, straight from the database. */
export interface PastIncident {
  date: string | null;
  engineer: string | null;
  symptom: string | null;
  root_cause: string | null;
  actions: string | null;
  method: FixMethod | null;
  parts_replaced: string | null;
  summary: string | null;
  session_id: number | null;
  similarity: number | null;
}

/** A question the assistant asked, with answers the technician can tap. */
export interface PendingQuestion {
  question: string;
  options: string[];
}

/** Structured extras saved with an agent message. */
export interface StepData {
  past_incidents?: PastIncident[];
  questions?: PendingQuestion[];
}

export interface FixRecord {
  engineer: string;
  root_cause: string;
  actions: string;
  method: FixMethod;
  parts_replaced?: string;
}

export interface FixDraft {
  root_cause: string;
  actions: string;
  parts_replaced: string;
}

export interface ChatMessage {
  role: "user" | "agent";
  content: string;
  images: string[];
  attachments?: Attachment[];
  step_data?: StepData | null;
  timestamp: string;
  type?: string;
  /** Set locally when a request fails, so the UI can style it as an error. */
  isError?: boolean;
}

export interface SessionSummary {
  id: number;
  machine_id: string | null;
  intent?: ChatIntent | null;
  title: string;
  timestamp: string;
  resolved: boolean;
}

export interface MaintenanceReport {
  sessionId: number;
  machineId: string | null;
  title: string;
  problemDescription: string;
  diagnosis: string;
  solutionSteps: string[];
  images: { url: string; caption: string }[];
  resolvedAt: string | null;
  resolution?: (FixRecord & { symptom?: string | null; summary?: string; resolved_at?: string }) | null;
  timestamp: string;
}

interface AssistantState {
  sessions: SessionSummary[];
  activeSessionId: number | null;
  messages: ChatMessage[];
  selectedMachineId: string | null;
  isAsking: boolean;
  isResolving: boolean;
  contextSource: string | null;
  error: string | null;
  report: MaintenanceReport | null;
  /** null until the user picks a purpose for the current chat. */
  intent: ChatIntent | null;
}

const initialState: AssistantState = {
  sessions: [],
  activeSessionId: null,
  messages: [],
  selectedMachineId: null,
  isAsking: false,
  isResolving: false,
  contextSource: null,
  error: null,
  report: null,
  intent: null,
};

interface AskResponse {
  role: "agent";
  content: string;
  session_id: number;
  machine_id: string | null;
  manual_id: string | null;
  images: string[];
  attachments: Attachment[];
  past_incidents?: PastIncident[];
  questions?: PendingQuestion[];
  mode: AskMode;
  intent: ChatIntent | null;
  context_source: string;
  timestamp: string;
}

export const askAssistant = createAsyncThunk<
  AskResponse,
  { query: string; mode?: AskMode },
  { state: { assistant: AssistantState } }
>("assistant/ask", async ({ query, mode = "answer" }, { getState }) => {
  const { activeSessionId, selectedMachineId, intent } = getState().assistant;
  return apiFetch<AskResponse>("/api/assistant", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      mode,
      session_id: activeSessionId ?? undefined,
      machine_id: selectedMachineId ?? undefined,
      // Only sent on the first turn; afterwards the session carries it server-side.
      intent: activeSessionId ? undefined : intent ?? undefined,
    }),
  });
});

interface AskWithImageResponse {
  role: "agent";
  content: string;
  session_id: number;
  machine_id: string | null;
  manual_id: string | null;
  user_image: string | null;
  intent: ChatIntent | null;
  context_source: string;
  timestamp: string;
}

/**
 * Ask about a photo or diagram attached directly in chat — a technician's own
 * phone photo of a leak or a damaged part, not one of the manual's own figures
 * (those already flow through ingestion and ordinary retrieval).
 *
 * A separate endpoint/thunk from askAssistant rather than an optional image on
 * it: the backend can't mix a JSON body with a file upload on one route, so
 * this posts FormData instead.
 */
export const askAssistantWithImage = createAsyncThunk<
  AskWithImageResponse,
  { query: string; file: File },
  { state: { assistant: AssistantState } }
>("assistant/askWithImage", async ({ query, file }, { getState }) => {
  const { activeSessionId, selectedMachineId, intent } = getState().assistant;
  const form = new FormData();
  form.append("query", query);
  form.append("image", file);
  if (activeSessionId) form.append("session_id", String(activeSessionId));
  if (selectedMachineId) form.append("machine_id", selectedMachineId);
  // Only sent on the first turn; afterwards the session carries it server-side.
  if (!activeSessionId && intent) form.append("intent", intent);

  return apiFetch<AskWithImageResponse>("/api/assistant/ask-with-image", {
    method: "POST",
    body: form,
  });
});

export const fetchSessions = createAsyncThunk("assistant/fetchSessions", async () =>
  apiFetch<SessionSummary[]>("/api/assistant/sessions")
);

export const openSession = createAsyncThunk(
  "assistant/openSession",
  async (sessionId: number) => {
    const history = await apiFetch<ChatMessage[]>(
      `/api/assistant/sessions/${sessionId}/history`
    );
    return { sessionId, history };
  }
);

export const deleteSession = createAsyncThunk(
  "assistant/deleteSession",
  async (sessionId: number) => {
    await apiFetch(`/api/assistant/sessions/${sessionId}`, { method: "DELETE" });
    return sessionId;
  }
);

/** Files what actually fixed the machine so future reports of the same fault
 *  on this machine show it first. */
export const resolveSession = createAsyncThunk(
  "assistant/resolve",
  async ({ sessionId, fix }: { sessionId: number; fix: FixRecord }) =>
    apiFetch<{ status: string; summary: string; resolution: FixRecord }>(
      `/api/assistant/sessions/${sessionId}/resolve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fix),
      }
    )
);

/** A draft of the fix record from the conversation, for the engineer to correct.
 *  Never includes the engineer's name or the method — only they know those. */
export const fetchResolveDraft = createAsyncThunk(
  "assistant/resolveDraft",
  async (sessionId: number) =>
    apiFetch<FixDraft>(`/api/assistant/sessions/${sessionId}/resolve-draft`, {
      method: "POST",
    })
);

export const fetchReport = createAsyncThunk(
  "assistant/fetchReport",
  async (sessionId: number) =>
    apiFetch<MaintenanceReport>(`/api/assistant/sessions/${sessionId}/report`)
);

const assistantSlice = createSlice({
  name: "assistant",
  initialState,
  reducers: {
    setSelectedMachine(state, action: PayloadAction<string | null>) {
      state.selectedMachineId = action.payload;
    },
    setIntent(state, action: PayloadAction<ChatIntent | null>) {
      state.intent = action.payload;
    },
    startNewSession(state) {
      state.activeSessionId = null;
      state.messages = [];
      state.contextSource = null;
      state.error = null;
      state.report = null;
      // Cleared so the next chat asks its purpose again rather than silently
      // inheriting the previous conversation's mode.
      state.intent = null;
    },
    clearError(state) {
      state.error = null;
    },
    clearReport(state) {
      state.report = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(askAssistant.pending, (state, action) => {
        state.isAsking = true;
        state.error = null;
        // Echo the question immediately so the conversation feels responsive.
        state.messages.push({
          role: "user",
          content: action.meta.arg.query,
          images: [],
          timestamp: new Date().toISOString(),
        });
      })
      .addCase(askAssistant.fulfilled, (state, action) => {
        state.isAsking = false;
        state.activeSessionId = action.payload.session_id;
        state.contextSource = action.payload.context_source;
        if (action.payload.intent) state.intent = action.payload.intent;
        state.messages.push({
          role: "agent",
          content: action.payload.content,
          images: action.payload.images || [],
          attachments: action.payload.attachments || [],
          // Same shape session history returns, so a live reply and a reopened
          // chat render through one path.
          step_data: {
            past_incidents: action.payload.past_incidents || [],
            questions: action.payload.questions || [],
          },
          timestamp: action.payload.timestamp,
          type: action.payload.mode === "wizard" ? "wizard_step" : "text",
        });
      })
      .addCase(askAssistant.rejected, (state, action) => {
        state.isAsking = false;
        state.error = action.error.message || "Request failed";
        state.messages.push({
          role: "agent",
          content: `Could not answer that: ${state.error}`,
          images: [],
          timestamp: new Date().toISOString(),
          isError: true,
        });
      })
      .addCase(askAssistantWithImage.pending, (state, action) => {
        state.isAsking = true;
        state.error = null;
        // Local preview via object URL while the upload is in flight — swapped
        // for nothing on success since the server echoes the same image back
        // in session history on reload anyway.
        state.messages.push({
          role: "user",
          content: action.meta.arg.query,
          images: [URL.createObjectURL(action.meta.arg.file)],
          timestamp: new Date().toISOString(),
        });
      })
      .addCase(askAssistantWithImage.fulfilled, (state, action) => {
        state.isAsking = false;
        state.activeSessionId = action.payload.session_id;
        state.contextSource = action.payload.context_source;
        if (action.payload.intent) state.intent = action.payload.intent;
        state.messages.push({
          role: "agent",
          content: action.payload.content,
          images: [],
          timestamp: action.payload.timestamp,
        });
      })
      .addCase(askAssistantWithImage.rejected, (state, action) => {
        state.isAsking = false;
        state.error = action.error.message || "Image analysis failed";
        state.messages.push({
          role: "agent",
          content: `Could not analyze that image: ${state.error}`,
          images: [],
          timestamp: new Date().toISOString(),
          isError: true,
        });
      })
      .addCase(fetchSessions.fulfilled, (state, action) => {
        state.sessions = action.payload;
      })
      .addCase(openSession.fulfilled, (state, action) => {
        state.activeSessionId = action.payload.sessionId;
        state.messages = action.payload.history.map((m) => ({
          ...m,
          images: m.images || [],
        }));
        const session = state.sessions.find((s) => s.id === action.payload.sessionId);
        state.selectedMachineId = session?.machine_id ?? state.selectedMachineId;
        // Reopening a thread resumes the purpose it was started with, so a
        // learning conversation doesn't turn into a fault report mid-way.
        state.intent = session?.intent ?? null;
        state.report = null;
      })
      .addCase(deleteSession.fulfilled, (state, action) => {
        state.sessions = state.sessions.filter((s) => s.id !== action.payload);
        if (state.activeSessionId === action.payload) {
          state.activeSessionId = null;
          state.messages = [];
        }
      })
      .addCase(resolveSession.pending, (state) => {
        state.isResolving = true;
      })
      .addCase(resolveSession.fulfilled, (state, action) => {
        state.isResolving = false;
        const session = state.sessions.find((s) => s.id === action.meta.arg.sessionId);
        if (session) session.resolved = true;
        state.messages.push({
          role: "agent",
          content: `**Fix recorded.** The next time this fault is reported on this machine, this fix is shown first.\n\n${action.payload.summary}`,
          images: [],
          timestamp: new Date().toISOString(),
        });
      })
      .addCase(resolveSession.rejected, (state, action) => {
        state.isResolving = false;
        state.error = action.error.message || "Could not record the fix";
      })
      .addCase(fetchReport.fulfilled, (state, action) => {
        state.report = action.payload;
      })
      .addCase(fetchReport.rejected, (state, action) => {
        state.error = action.error.message || "Could not build the report";
      });
  },
});

export const { setSelectedMachine, setIntent, startNewSession, clearError, clearReport } =
  assistantSlice.actions;
export default assistantSlice.reducer;

import { createSlice, createAsyncThunk, PayloadAction } from "@reduxjs/toolkit";
import { apiFetch } from "../api";

export type AskMode = "answer" | "wizard";

export interface ChatMessage {
  role: "user" | "agent";
  content: string;
  images: string[];
  timestamp: string;
  type?: string;
  /** Set locally when a request fails, so the UI can style it as an error. */
  isError?: boolean;
}

export interface SessionSummary {
  id: number;
  machine_id: string | null;
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
};

interface AskResponse {
  role: "agent";
  content: string;
  session_id: number;
  machine_id: string | null;
  manual_id: string | null;
  images: string[];
  mode: AskMode;
  context_source: string;
  timestamp: string;
}

export const askAssistant = createAsyncThunk<
  AskResponse,
  { query: string; mode?: AskMode },
  { state: { assistant: AssistantState } }
>("assistant/ask", async ({ query, mode = "answer" }, { getState }) => {
  const { activeSessionId, selectedMachineId } = getState().assistant;
  return apiFetch<AskResponse>("/api/assistant", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      mode,
      session_id: activeSessionId ?? undefined,
      machine_id: selectedMachineId ?? undefined,
    }),
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

/** Files what actually fixed the machine so future questions surface it. */
export const resolveSession = createAsyncThunk(
  "assistant/resolve",
  async ({ sessionId, operatorFix }: { sessionId: number; operatorFix: string }) =>
    apiFetch<{ status: string; summary: string }>(
      `/api/assistant/sessions/${sessionId}/resolve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ operator_fix: operatorFix }),
      }
    )
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
    startNewSession(state) {
      state.activeSessionId = null;
      state.messages = [];
      state.contextSource = null;
      state.error = null;
      state.report = null;
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
        state.messages.push({
          role: "agent",
          content: action.payload.content,
          images: action.payload.images || [],
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
        const session = state.sessions.find((s) => s.id === state.activeSessionId);
        if (session) session.resolved = true;
        state.messages.push({
          role: "agent",
          content: `**Fix recorded.** Future questions about this machine will surface it.\n\n${action.payload.summary}`,
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

export const { setSelectedMachine, startNewSession, clearError, clearReport } =
  assistantSlice.actions;
export default assistantSlice.reducer;

import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { apiFetch } from "../api";

export interface Machine {
  machine_id: string;
  name: string;
  location: string;
  manual_id: string;
  /** 0 means the linked manual has no ingested content yet. */
  manual_chunks: number;
}

interface MachinesState {
  items: Machine[];
  isSaving: boolean;
  status: string | null;
  error: string | null;
}

const initialState: MachinesState = {
  items: [],
  isSaving: false,
  status: null,
  error: null,
};

export const fetchMachines = createAsyncThunk("machines/fetch", async () =>
  apiFetch<Machine[]>("/api/machines")
);

export const saveMachine = createAsyncThunk(
  "machines/save",
  async (payload: Omit<Machine, "manual_chunks">) =>
    apiFetch<Machine>("/api/machines", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
);

export const deleteMachine = createAsyncThunk(
  "machines/delete",
  async (machineId: string) => {
    await apiFetch(`/api/machines/delete/${encodeURIComponent(machineId)}`, {
      method: "POST",
    });
    return machineId;
  }
);

const machinesSlice = createSlice({
  name: "machines",
  initialState,
  reducers: {
    clearStatus(state) {
      state.status = null;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchMachines.fulfilled, (state, action) => {
        state.items = action.payload;
      })
      .addCase(saveMachine.pending, (state) => {
        state.isSaving = true;
        state.error = null;
      })
      .addCase(saveMachine.fulfilled, (state, action) => {
        state.isSaving = false;
        const idx = state.items.findIndex(
          (m) => m.machine_id === action.payload.machine_id
        );
        if (idx >= 0) state.items[idx] = action.payload;
        else state.items.push(action.payload);
        state.status =
          action.payload.manual_chunks === 0
            ? `Saved. Note: manual "${action.payload.manual_id}" has no content yet — upload it under Manuals.`
            : `Saved ${action.payload.machine_id}.`;
      })
      .addCase(saveMachine.rejected, (state, action) => {
        state.isSaving = false;
        state.error = action.error.message || "Could not save the machine";
      })
      .addCase(deleteMachine.fulfilled, (state, action) => {
        state.items = state.items.filter((m) => m.machine_id !== action.payload);
        state.status = `Removed ${action.payload}.`;
      })
      .addCase(deleteMachine.rejected, (state, action) => {
        state.error = action.error.message || "Could not remove the machine";
      });
  },
});

export const { clearStatus } = machinesSlice.actions;
export default machinesSlice.reducer;

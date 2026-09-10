import { configureStore } from "@reduxjs/toolkit";
import assistantReducer from "./slices/assistantSlice";
import manualsReducer from "./slices/manualsSlice";
import machinesReducer from "./slices/machinesSlice";

export const store = configureStore({
  reducer: {
    assistant: assistantReducer,
    manuals: manualsReducer,
    machines: machinesReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

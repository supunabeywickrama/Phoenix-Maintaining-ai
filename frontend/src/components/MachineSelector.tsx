"use client";

import { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { AppDispatch, RootState } from "../store/store";
import { fetchMachines } from "../store/slices/machinesSlice";
import { setSelectedMachine } from "../store/slices/assistantSlice";
import { Cog, AlertTriangle } from "lucide-react";

export default function MachineSelector() {
  const dispatch = useDispatch<AppDispatch>();
  const machines = useSelector((s: RootState) => s.machines.items);
  const selected = useSelector((s: RootState) => s.assistant.selectedMachineId);

  useEffect(() => {
    dispatch(fetchMachines());
  }, [dispatch]);

  const current = machines.find((m) => m.machine_id === selected);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900 px-3 py-2">
        <Cog size={16} className="shrink-0 text-orange-400" />
        <select
          aria-label="Machine"
          value={selected ?? ""}
          onChange={(e) => dispatch(setSelectedMachine(e.target.value || null))}
          className="w-full bg-transparent text-sm font-medium text-white outline-none"
        >
          <option value="" className="bg-slate-900">
            No machine selected — general answers
          </option>
          {machines.map((m) => (
            <option key={m.machine_id} value={m.machine_id} className="bg-slate-900">
              {m.name} ({m.machine_id})
            </option>
          ))}
        </select>
      </div>

      {current && current.manual_chunks === 0 && (
        <p className="flex items-center gap-1.5 text-[11px] font-medium text-amber-400">
          <AlertTriangle size={12} />
          No manual content for {current.manual_id} yet — answers will be generic.
        </p>
      )}
    </div>
  );
}

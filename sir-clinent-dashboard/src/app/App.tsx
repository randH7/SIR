import { TopBar } from "../features/topbar/TopBar";
import { IncidentMap } from "../features/incident-map/IncidentMap";
import { PredictionPanel } from "../features/prediction-panel/PredictionPanel";

export function App() {
  return (
    <div className="h-full bg-neutral-950 text-neutral-100">
      <TopBar />

      <div className="h-[calc(100%-64px)] grid grid-cols-12 gap-3 p-3">
        <div className="col-span-9 rounded-2xl overflow-hidden border border-neutral-800 bg-neutral-900">
          <IncidentMap />
        </div>

        <div className="col-span-3 rounded-2xl border border-neutral-800 bg-neutral-900 overflow-hidden">
          <PredictionPanel />
        </div>
      </div>
    </div>
  );
}

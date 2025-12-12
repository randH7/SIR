import { dataset } from "../../shared/lib/dummyData";

export function PredictionPanel() {
  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-neutral-800">
        <div className="text-sm font-semibold">Predictions</div>
        <div className="text-xs text-neutral-400">
          Next 1–2 hours (demo data)
        </div>
      </div>

      <div className="p-3 space-y-3 overflow-auto">
        {dataset.predictions.map((p) => (
          <div
            key={p.id}
            className="rounded-2xl border border-neutral-800 bg-neutral-950/40 p-3"
          >
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">
                {p.theme.replaceAll("_", " ")}
              </div>
              <div className="text-xs text-neutral-300">
                {(p.confidence * 100).toFixed(0)}%
              </div>
            </div>

            <div className="mt-2 text-xs text-neutral-400">
              Radius: {Math.round(p.geoArea.radiusMeters)}m
            </div>

            <div className="mt-1 text-xs text-neutral-500">
              {new Date(p.timeWindow.start).toLocaleTimeString()} →{" "}
              {new Date(p.timeWindow.end).toLocaleTimeString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

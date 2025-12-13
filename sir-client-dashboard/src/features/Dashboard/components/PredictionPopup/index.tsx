import { Popup } from "react-map-gl/maplibre";

import { PredictionPopupProps } from "./type";
import { Spinner } from "../Spinner";

const riskChip = (risk: string) => {
  switch (risk) {
    case "low":
      return "border-emerald-500/30 bg-emerald-500/10 text-emerald-200";
    case "medium":
      return "border-amber-500/30 bg-amber-500/10 text-amber-200";
    case "high":
      return "border-orange-500/30 bg-orange-500/10 text-orange-200";
    case "extreme":
      return "border-red-500/30 bg-red-500/10 text-red-200";
    default:
      return "border-neutral-700 bg-neutral-900 text-neutral-200";
  }
};

const pretty = (s: string) => s.replaceAll("_", " ");

export function PredictionPopup({
  prediction,
  details,
  isLoading,
  onClose,
}: PredictionPopupProps) {
  const start = new Date(prediction.window_start).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  const end = new Date(prediction.window_end).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  const topExplanations =
    details?.explainability.explanations
      ?.slice()
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 3) ?? [];

  return (
    <Popup
      latitude={prediction.center_lat}
      longitude={prediction.center_lng}
      closeButton={false}
      closeOnClick={false}
      anchor="bottom"
      offset={14}
      onClose={onClose}
      className="sir-popup"
    >
      <div className="min-w-[260px] rounded-2xl border border-neutral-800 bg-neutral-950/95 text-neutral-100 shadow-xl">
        <div className="flex items-start justify-between gap-3 p-3 border-b border-neutral-800">
          <div className="min-w-0">
            <div className="text-sm font-semibold truncate">
              {pretty(prediction.type)}
            </div>
            <div className="mt-1 text-xs text-neutral-400 capitalize">
              {pretty(prediction.theme)}
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="shrink-0 h-7 w-7 rounded-lg border border-neutral-800 bg-neutral-900 hover:bg-neutral-800 transition grid place-items-center"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="p-3 space-y-3">
          <div className="flex items-center justify-between">
            <span
              className={[
                "text-[11px] px-2 py-1 rounded-lg border font-medium capitalize",
                riskChip(prediction.risk_level),
              ].join(" ")}
            >
              {pretty(prediction.risk_level)}
            </span>

            <div className="text-xs text-neutral-300">
              {Math.round(prediction.confidence * 100)}%
            </div>
          </div>

          <div className="text-xs text-neutral-400">
            <div className="text-neutral-500">Window</div>
            <div className="text-neutral-200">
              {start} → {end}
            </div>
          </div>

          <div className="text-xs text-neutral-400">
            <div className="text-neutral-500">Radius</div>
            <div className="text-neutral-200">
              {Math.round(prediction.radius_meters)}m
            </div>
          </div>

          {prediction.summary ? (
            <div className="text-xs text-neutral-300">{prediction.summary}</div>
          ) : null}
        </div>

        <div className="p-3">
          {isLoading ? (
            <div className="py-1">
              <Spinner size={18} />
            </div>
          ) : details ? (
            <div className="space-y-2">
              <div className="text-xs text-neutral-500">
                Why this area is flagged?
              </div>

              {topExplanations.length === 0 ? (
                <div className="text-xs text-neutral-400">
                  No explainability available.
                </div>
              ) : (
                <div className="space-y-2">
                  {topExplanations.map((x, idx) => (
                    <div
                      key={`${x.signal_type}-${idx}`}
                      className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-xs font-medium text-neutral-200 capitalize">
                          {x.signal_type.replaceAll("_", " ")}
                        </div>
                        <div className="text-[11px] text-neutral-400">
                          weight {x.weight.toFixed(2)}
                        </div>
                      </div>

                      <div className="mt-1 text-xs text-neutral-300 line-clamp-2">
                        {x.summary}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="text-xs text-neutral-400">No details</div>
          )}
        </div>
      </div>
    </Popup>
  );
}

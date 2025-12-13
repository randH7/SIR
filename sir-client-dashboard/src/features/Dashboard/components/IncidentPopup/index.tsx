import { Popup } from "react-map-gl/maplibre";
import { IncidentPopupProps } from "./types";
import './style.css'

const statusChip = (status: string) => {
  switch (status) {
    case "open":
      return "border-emerald-500/30 bg-emerald-500/10 text-emerald-200";
    case "in_progress":
      return "border-blue-500/30 bg-blue-500/10 text-blue-200";
    case "closed":
      return "border-neutral-500/30 bg-neutral-500/10 text-neutral-200";
    default:
      return "border-neutral-700 bg-neutral-900 text-neutral-200";
  }
};

const severityChip = (severity: number) => {
  if (severity >= 5) return "border-red-500/30 bg-red-500/10 text-red-200";
  if (severity === 4)
    return "border-orange-500/30 bg-orange-500/10 text-orange-200";
  if (severity === 3)
    return "border-amber-500/30 bg-amber-500/10 text-amber-200";
  if (severity === 2)
    return "border-emerald-500/30 bg-emerald-500/10 text-emerald-200";
  return "border-sky-500/30 bg-sky-500/10 text-sky-200";
};

export const IncidentPopup = ({ incident, onClose }: IncidentPopupProps) => {
  const time = new Date(incident.occurred_at).toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <Popup
      latitude={incident.lat}
      longitude={incident.lng}
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
              {incident.title}
            </div>
            <div className="mt-1 text-xs text-neutral-400 capitalize">
              {incident.domain.replaceAll("_", " ")}
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
          <div className="flex flex-wrap gap-2">
            <span
              className={[
                "text-[11px] px-2 py-1 rounded-lg border font-medium capitalize",
                statusChip(incident.status),
              ].join(" ")}
            >
              {incident.status.replaceAll("_", " ")}
            </span>

            <span
              className={[
                "text-[11px] px-2 py-1 rounded-lg border font-medium",
                severityChip(incident.severity),
              ].join(" ")}
            >
              Severity {incident.severity}
            </span>
          </div>

          <div className="text-xs text-neutral-400">
            <div className="text-neutral-500">Occurred</div>
            <div className="text-neutral-200">{time}</div>
          </div>

          {incident.source_channels?.length ? (
            <div className="text-xs text-neutral-400">
              <div className="text-neutral-500">Sources</div>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {incident.source_channels.map((s) => (
                  <span
                    key={s}
                    className="text-[11px] px-2 py-1 rounded-lg border border-neutral-800 bg-neutral-900"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </Popup>
  );
};

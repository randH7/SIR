import { IncidentsLayerProps } from "./types";
import { Marker } from "react-map-gl/maplibre";

const severityClass = (severity: number) => {
  if (severity >= 5) return "bg-red-500 ring-red-500/25";
  if (severity === 4) return "bg-orange-500 ring-orange-500/25";
  if (severity === 3) return "bg-amber-500 ring-amber-500/25";
  if (severity === 2) return "bg-emerald-500 ring-emerald-500/25";
  return "bg-sky-500 ring-sky-500/25";
};

const statusRingClass = (status: string) => {
  switch (status) {
    case "open":
      return "ring-2 ring-white/40";
    case "in_progress":
      return "ring-2 ring-blue-400/60";
    case "closed":
      return "ring-2 ring-neutral-500/60";
    default:
      return "ring-2 ring-white/25";
  }
};

export function IncidentsLayer({
  incidents,
  onSelect,
  selectedId,
}: IncidentsLayerProps) {
  return (
    <>
      {incidents.map((inc) => {
        const isSelected = selectedId === inc.incident_id;

        return (
          <Marker
            key={inc.incident_id}
            latitude={inc.lat}
            longitude={inc.lng}
            anchor="center"
            onClick={(e) => {
              // prevent the map click from also firing
              e.originalEvent.stopPropagation();
              onSelect?.(inc.incident_id);
            }}
          >
            <div
              className={[
                "h-3.5 w-3.5 rounded-full ring-4 transition",
                severityClass(inc.severity),
                isSelected ? "scale-125" : "hover:scale-110",
              ].join(" ")}
              title={inc.title}
              style={{ pointerEvents: "auto" }} // important
            />
          </Marker>
        );
      })}
    </>
  );
}
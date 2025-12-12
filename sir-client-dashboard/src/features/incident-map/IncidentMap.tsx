import Map, { Marker } from "react-map-gl/maplibre";
import { dataset } from "../../shared/lib/dummyData";
import { PredictionRingsLayer } from "./PredictionRingsLayer";

const OSM_RASTER_STYLE = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [
    {
      id: "osm",
      type: "raster",
      source: "osm",
    },
  ],
} as const;

export function IncidentMap() {
  return (
    <div className="h-full w-full">
      <Map
        initialViewState={{
          latitude: 24.7136,
          longitude: 46.6753,
          zoom: 11.5,
        }}
        mapStyle={OSM_RASTER_STYLE}
      >
        {/* 🔥 Predictions first (under markers) */}
        <PredictionRingsLayer predictions={dataset.predictions} />

        {/* Incidents on top */}
        {dataset.incidents.map((inc) => (
          <Marker
            key={inc.id}
            latitude={inc.location.lat}
            longitude={inc.location.lng}
            anchor="center"
          >
            <div className="h-3 w-3 rounded-full bg-red-500 ring-4 ring-red-500/25" />
          </Marker>
        ))}
      </Map>
      <div className="pointer-events-none absolute inset-0 bg-neutral-950/25" />
    </div>
  );
}


import MapGL from "react-map-gl/maplibre";
import type { MapRef } from "react-map-gl/maplibre";
import { useMemo, useRef, useState } from "react";

import { MapProps } from "./types";
import { IncidentsLayer } from "../IncidentsLayer";
import { IncidentPopup } from "../IncidentPopup";
import "../IncidentPopup/style.css";
import { PredictionLayer } from "../PredictionLayer";
import { PredictionPopup } from "../PredictionPopup";
import { usePredictionDetails } from "../../logic";

const STYLE_URL =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

export const Map = ({ incidents, predictions }: MapProps) => {
  const mapRef = useRef<MapRef>(null);

  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(
    null
  );
  const [selectedPredictionId, setSelectedPredictionId] = useState<
    string | null
  >(null);

  const selectedIncident = useMemo(
    () =>
      selectedIncidentId
        ? incidents.find((i) => i.incident_id === selectedIncidentId)
        : null,
    [incidents, selectedIncidentId]
  );

  const selectedPrediction = useMemo(
    () =>
      selectedPredictionId
        ? predictions.find((p) => p.prediction_id === selectedPredictionId) ??
          null
        : null,
    [predictions, selectedPredictionId]
  );

  const details = usePredictionDetails(selectedPredictionId);

  return (
    <div className="relative h-full w-full">
      <MapGL
        ref={mapRef}
        initialViewState={{ latitude: 24.7136, longitude: 46.6753, zoom: 11.5 }}
        mapStyle={STYLE_URL}
        interactiveLayerIds={["predictions-fill", "predictions-ring"]}
        onClick={(e) => {
          // Priority: if user clicked a prediction polygon, select it
          const feature = e.features?.[0];
          const predictionId = feature?.properties?.prediction_id as
            | string
            | undefined;

          if (predictionId) {
            setSelectedPredictionId(predictionId);
            setSelectedIncidentId(null);
            return;
          }

          // click on map background closes popups
          setSelectedPredictionId(null);
          setSelectedIncidentId(null);
        }}
        onLoad={() => {
          const map = mapRef.current?.getMap();
          if (!map) return;

          const style = map.getStyle();
          const layers = style?.layers ?? [];

          // 1) Prefer Arabic names (fallback to default `name`)
          // 2) Increase label sizes (streets + places)
          for (const layer of layers) {
            if (layer.type !== "symbol") continue;

            // only layers that actually draw text
            const hasTextField =
              typeof layer.layout?.["text-field"] !== "undefined";
            if (!hasTextField) continue;

            // Force Arabic (works when source has `name:ar`)
            map.setLayoutProperty(layer.id, "text-field", [
              "coalesce",
              ["get", "name:ar"],
              ["get", "name"],
            ]);

            // Make labels larger (you can tune these)
            // For street labels, bigger at higher zooms.
            map.setLayoutProperty(layer.id, "text-size", [
              "interpolate",
              ["linear"],
              ["zoom"],
              10,
              12,
              14,
              16,
              16,
              20,
            ]);

            // Optional: improve readability on dark maps
            map.setPaintProperty(
              layer.id,
              "text-color",
              "rgba(255, 255, 255, 0.9)"
            );
            map.setPaintProperty(
              layer.id,
              "text-halo-color",
              "rgba(0,0,0,0.8)"
            );
            map.setPaintProperty(layer.id, "text-halo-width", 1.2);
          }
        }}
      >
        {/* 🔥 Predictions under incidents */}
        <PredictionLayer
          predictions={predictions}
          selectedId={selectedPredictionId}
        />

        {/* Incidents on top */}
        <IncidentsLayer
          incidents={incidents}
          selectedId={selectedIncidentId}
          onSelect={setSelectedIncidentId}
        />

        {selectedIncident ? (
          <IncidentPopup
            incident={selectedIncident}
            onClose={() => setSelectedIncidentId(null)}
          />
        ) : null}

        {selectedPrediction ? (
          <PredictionPopup
            prediction={selectedPrediction}
            details={details.data}
            isLoading={details.isLoading}
            onClose={() => setSelectedPredictionId(null)}
          />
        ) : null}
      </MapGL>
    </div>
  );
};

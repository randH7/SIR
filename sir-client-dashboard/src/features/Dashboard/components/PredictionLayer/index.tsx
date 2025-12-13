import { Source, Layer } from "react-map-gl/maplibre";

import { predictionsToCircleGeoJson } from "./logic";
import { PredictionLayerProps } from "./types";

export const PredictionLayer = ({ predictions, selectedId }: PredictionLayerProps) => {
  const data = predictionsToCircleGeoJson(predictions);

  return (
    <Source id="predictions" type="geojson" data={data}>
      <Layer
        id="predictions-fill"
        type="fill"
        paint={{
          "fill-color": [
            "match",
            ["get", "risk_level"],
            "low",
            "rgb(16,185,129)",
            "medium",
            "rgb(245,158,11)",
            "high",
            "rgb(249,115,22)",
            "extreme",
            "rgb(239,68,68)",
            "rgb(255,80,80)",
          ],
          // Selected zone slightly stronger
          "fill-opacity": [
            "case",
            ["==", ["get", "prediction_id"], selectedId ?? ""],
            ["*", ["coalesce", ["get", "alpha"], 0.12], 1.5],
            ["coalesce", ["get", "alpha"], 0.12],
          ],
        }}
      />

      {/* <Layer
        id="predictions-ring"
        type="line"
        filter={["==", ["get", "ring_t"], 1]}
        paint={{
          "line-color": [
            "match",
            ["get", "risk_level"],
            "low",
            "rgba(16,185,129,0.75)",
            "medium",
            "rgba(245,158,11,0.75)",
            "high",
            "rgba(249,115,22,0.8)",
            "extreme",
            "rgba(239,68,68,0.9)",
            "rgba(255,120,120,0.85)",
          ],
          "line-width": [
            "case",
            ["==", ["get", "prediction_id"], selectedId ?? ""],
            3,
            2,
          ],
        }}
      /> */}
    </Source>
  );
};

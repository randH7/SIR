import { Source, Layer } from "react-map-gl/maplibre";
import type { Prediction } from "../../shared/types/schemas";
import { predictionsToGeoJson } from "./mapGeo";

type Props = {
  predictions: Prediction[];
};

export function PredictionRingsLayer({ predictions }: Props) {
  const data = predictionsToGeoJson(predictions);

  return (
    <Source id="predictions" type="geojson" data={data}>
      {/* soft filled circle */}
      <Layer
        id="predictions-fill"
        type="circle"
        paint={{
          // radius in pixels; we scale meters roughly by zoom using a step function
          // (good enough for demo, later we’ll do accurate meters via a circle polygon)
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["zoom"],
            9,
            ["*", ["get", "radiusMeters"], 0.03],
            12,
            ["*", ["get", "radiusMeters"], 0.08],
            15,
            ["*", ["get", "radiusMeters"], 0.18],
          ],
          "circle-color": "rgba(255, 80, 80, 0.35)",
          "circle-blur": 0.7,
        }}
      />

      {/* ring outline */}
      <Layer
        id="predictions-ring"
        type="circle"
        paint={{
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["zoom"],
            9,
            ["*", ["get", "radiusMeters"], 0.03],
            12,
            ["*", ["get", "radiusMeters"], 0.08],
            15,
            ["*", ["get", "radiusMeters"], 0.18],
          ],
          "circle-color": "rgba(0,0,0,0)",
          "circle-stroke-color": "rgba(255, 120, 120, 0.85)",
          "circle-stroke-width": 2,
        }}
      />
    </Source>
  );
}

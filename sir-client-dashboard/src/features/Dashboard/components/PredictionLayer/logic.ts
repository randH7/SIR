import type { FeatureCollection, Feature } from "geojson";
import { circle } from "@turf/turf";

import { Predictions } from "./types";

const RING_STOPS = [
  { t: 1.0, alpha: 0.07 }, // center strongest
  { t: 0.75, alpha: 0.1 },
  { t: 0.5, alpha: 0.2 },
  { t: 0.25, alpha: 0.2 }, // edge light
];

export const predictionsToCircleGeoJson = (
  predictions: Predictions
): FeatureCollection => {
  const features: Feature[] = [];

  for (const p of predictions) {
    const radiusKm = p.radius_meters / 1000;

    for (const stop of RING_STOPS) {
      const poly = circle([p.center_lng, p.center_lat], radiusKm * stop.t, {
        steps: 64,
        units: "kilometers",
      });

      poly.properties = {
        prediction_id: p.prediction_id,
        type: p.type,
        theme: p.theme,
        confidence: p.confidence,
        risk_level: p.risk_level,
        window_start: p.window_start,
        window_end: p.window_end,
        radius_meters: p.radius_meters,
        alpha: stop.alpha, // 🔥 per-ring opacity
        ring_t: stop.t, // optional debug
      };

      features.push(poly as Feature);
    }
  }

  return { type: "FeatureCollection", features };
};

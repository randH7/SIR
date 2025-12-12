import type { Prediction } from "../../shared/types/schemas";

type FeatureCollection = GeoJSON.FeatureCollection<GeoJSON.Geometry>;

export function predictionsToGeoJson(
  predictions: Prediction[]
): FeatureCollection {
  return {
    type: "FeatureCollection",
    features: predictions.map((p) => ({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [p.geoArea.center.lng, p.geoArea.center.lat],
      },
      properties: {
        id: p.id,
        theme: p.theme,
        confidence: p.confidence,
        radiusMeters: p.geoArea.radiusMeters,
      },
    })),
  };
}

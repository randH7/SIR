import { apiClient } from "../../apiClient";
import { ApiClientSchema } from "../../apiClient.types";
import { PredictionsResponseSchema, PredictionDetailsSchema } from "./types";

const ListSchema = ApiClientSchema(PredictionsResponseSchema);
const DetailsSchema = ApiClientSchema(PredictionDetailsSchema);

export async function getPredictions(params: {
  cityId: string;
  horizonMinutes: number;
  bbox?: string;
  minConfidence?: number;
}) {
  const res = await apiClient.get("/api/v1/predictions", {
    params: {
      city_id: params.cityId,
      horizon_minutes: params.horizonMinutes,
      bbox: params.bbox,
      min_confidence: params.minConfidence,
    },
  });

  const parsed = ListSchema.parse(res.data);
  return (
    parsed.data ?? { predictions: [], horizon_minutes: params.horizonMinutes }
  );
}

export async function getPredictionDetails(predictionId: string) {
  const res = await apiClient.get(`/api/v1/predictions/${predictionId}`);
  const parsed = DetailsSchema.parse(res.data);
  return parsed.data;
}

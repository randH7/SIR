import z from "zod";
import { PredictionsResponseSchema } from "../../../../services/api/predictions/types";

type PredictionResponse = z.infer<typeof PredictionsResponseSchema>;

export type Predictions = PredictionResponse["predictions"];

export type PredictionItem = PredictionResponse["predictions"][number];

export interface PredictionLayerProps {
  predictions: PredictionItem[];
  selectedId?: string | null;
}

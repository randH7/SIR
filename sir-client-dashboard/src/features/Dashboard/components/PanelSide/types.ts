import { z } from "zod";
import { PredictionsResponseSchema } from "../../../../services/api/predictions/types";

export type PredictionsResponse = z.infer<typeof PredictionsResponseSchema>;
export type PredictionItem = PredictionsResponse['predictions'][number];

export interface PanelSideProps {
  nextHours: PredictionsResponse["horizon_minutes"];
  items: PredictionsResponse["predictions"];
}

export type PanelSummaryItem = {
  type: PredictionItem["type"];
  title: string;
  subtitle: string; // e.g. "3 zones flagged"
  riskLevel?: PredictionItem["risk_level"];
};

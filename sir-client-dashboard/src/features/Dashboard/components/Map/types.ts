import { Incidents } from "../IncidentsLayer/types";
import { Predictions } from "../PredictionLayer/types";

export interface MapProps {
  incidents: Incidents;
  predictions: Predictions;
}

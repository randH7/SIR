import z from "zod";
import { PredictionDetailsSchema } from "../../../../services/api/predictions/types";
import { PredictionItem } from "../PredictionLayer/types";

type PredictionDetails = z.infer<typeof PredictionDetailsSchema>;

export interface PredictionPopupProps {
  prediction: PredictionItem;
  details?: PredictionDetails | null;
  isLoading: boolean;
  onClose: () => void;
}

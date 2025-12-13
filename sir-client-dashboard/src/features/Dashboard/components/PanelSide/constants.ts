import { PredictionItem } from "./types";

export const RISK_RANK: Record<PredictionItem["risk_level"], number> = {
  low: 1,
  medium: 2,
  high: 3,
  extreme: 4,
};

export const TYPE_META: Record<
  PredictionItem["type"],
  { title: string; unit: string; showRisk: boolean }
> = {
  current_hotspots: {
    title: "Current Hotspots",
    unit: "zones",
    showRisk: false,
  },
  congestion_waves: {
    title: "Congestion Waves",
    unit: "areas",
    showRisk: true,
  },
  crime_clusters: {
    title: "Crime Clusters",
    unit: "clusters",
    showRisk: false,
  },
  overcrowding_risks: {
    title: "Overcrowding Risks",
    unit: "areas",
    showRisk: true,
  },
};

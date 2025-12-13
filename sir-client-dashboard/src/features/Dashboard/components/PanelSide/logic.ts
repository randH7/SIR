import { TYPE_META, RISK_RANK } from "./constants";
import type { PanelSummaryItem, PredictionItem } from "./types";

export const buildPanelSummary = (
  predictions: PredictionItem[]
): PanelSummaryItem[] => {
  console.log({predictions});
  const byType = new Map<PredictionItem["type"], PredictionItem[]>();

  for (const p of predictions) {
    const list = byType.get(p.type) ?? [];
    list.push(p);
    byType.set(p.type, list);
  }

  const order: PredictionItem["type"][] = [
    "current_hotspots",
    "congestion_waves",
    "crime_clusters",
    "overcrowding_risks",
  ];

  return order.map((type) => {
    const meta = TYPE_META[type];
    const list = byType.get(type) ?? [];
    const count = list.length;

    const maxRisk =
      list.length === 0
        ? undefined
        : list.reduce((best, cur) =>
            RISK_RANK[cur.risk_level] > RISK_RANK[best.risk_level] ? cur : best
          ).risk_level;

    return {
      type,
      title: meta.title,
      subtitle: `${count} ${meta.unit} flagged`,
      riskLevel: meta.showRisk ? maxRisk : undefined,
    };
  });
};

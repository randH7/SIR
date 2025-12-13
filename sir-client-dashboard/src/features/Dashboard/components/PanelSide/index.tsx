import type { PanelSideProps } from "./types";
import { buildPanelSummary } from "./logic";

const riskChipClass = (risk: string) => {
  switch (risk) {
    case "low":
      return "border-emerald-500/30 bg-emerald-500/10 text-emerald-200";
    case "medium":
      return "border-amber-500/30 bg-amber-500/10 text-amber-200";
    case "extreme":
      return "border-red-500/40 bg-red-500/10 text-red-200 shadow-[0_0_0_1px_rgba(239,68,68,0.15)]";
    case "high":
      return "border-orange-500/40 bg-orange-500/10 text-orange-200 shadow-[0_0_0_1px_rgba(249,115,22,0.15)]";
    default:
      return "border-neutral-700 bg-neutral-900 text-neutral-200";
  }
};

const prettyRisk = (risk: string) => risk.replaceAll("_", " ");

export const PanelSide = ({ nextHours, items }: PanelSideProps) => {
  const minutes = nextHours;
  const hours = Math.round((minutes / 60) * 10) / 10;

  const summary = buildPanelSummary(items);

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-neutral-800">
        <div className="text-sm font-semibold">Prediction Layers</div>
        <div className="text-xs text-neutral-400">
          Next {minutes} min {minutes >= 60 ? `(${hours}h)` : ""}
        </div>
      </div>

      <div className="p-3 space-y-3 overflow-auto">
        {summary.map((card) => (
          <div
            key={card.type}
            className="rounded-2xl border border-neutral-800 bg-neutral-950/40 p-4"
          >
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold">{card.title}</div>
              {card.riskLevel ? (
                <div
                  className={[
                    "text-[11px] px-2 py-1 rounded-lg border font-medium capitalize",
                    "backdrop-blur-sm",
                    riskChipClass(card.riskLevel),
                  ].join(" ")}
                >
                  {prettyRisk(card.riskLevel)}
                </div>
              ) : null}
            </div>

            <div className="mt-2 text-sm text-neutral-200">{card.subtitle}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

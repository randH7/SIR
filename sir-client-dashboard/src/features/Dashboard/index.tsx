import { useState } from "react";

import { Map } from "./components/Map";
import { PanelSide } from "./components/PanelSide";
import { TopBar } from "./components/TopBar";

import { useGetData } from "./logic";

export const Dashboard = () => {
  const [horizonMinutes] = useState(700);
  const {
    bboxParam,
    bootstrapData,
    cityName,
    incidentsData,
    predictionsData,
    isLoading,
    isError,
  } = useGetData(horizonMinutes);

  // Page State
  if (isLoading) return <div className="p-6">Loading config…</div>;
  if (isError) return <div className="p-6">Failed to load config.</div>;
  if (!bootstrapData) return <div className="p-6">No config.</div>;

  return (
    <div className="h-full bg-neutral-950 text-neutral-100">
      <TopBar city={cityName} />

      <div className="h-[calc(100%-64px)] grid grid-cols-12 gap-3 p-3">
        <div className="col-span-9 rounded-2xl overflow-hidden border border-neutral-800 bg-neutral-900">
          <Map incidents={incidentsData} predictions={predictionsData} />
        </div>

        <div className="col-span-3 rounded-2xl border border-neutral-800 bg-neutral-900 overflow-hidden">
          <PanelSide nextHours={horizonMinutes} items={predictionsData} />
        </div>
      </div>
    </div>
  );
};

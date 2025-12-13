import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { getBootstrap } from "../../services/api/bootstrap";
import { BBox } from "./types";
import { getIncidents } from "../../services/api/incidents";
import {
  getPredictionDetails,
  getPredictions,
} from "../../services/api/predictions";
import { getLookupValue } from "../../utils/transformLookup";

export const cityId = "riyadh";

const bboxToParam = (b: BBox) =>
  `${b.min_lng},${b.min_lat},${b.max_lng},${b.max_lat}`; // minLng,minLat,maxLng,maxLat

export const useGetData = (horizonMinutes: number) => {
  const bootstrapQ = useQuery({
    queryKey: ["bootstrap", cityId],
    queryFn: () => getBootstrap(cityId),
  });

  const bboxParam = useMemo(() => {
    const bbox = bootstrapQ.data?.config.city.bbox;
    return bbox ? bboxToParam(bbox) : undefined;
  }, [bootstrapQ.data]);

  const incidentsQ = useQuery({
    queryKey: ["incidents", cityId, bboxParam],
    enabled: Boolean(bboxParam),
    queryFn: () => getIncidents({ cityId, bbox: bboxParam }),
    refetchInterval: bootstrapQ.data?.config.refresh_seconds.incidents
      ? bootstrapQ.data.config.refresh_seconds.incidents * 1000
      : 15000,
  });

  const predictionsQ = useQuery({
    queryKey: ["predictions", cityId, horizonMinutes, bboxParam],
    enabled: Boolean(bboxParam),
    queryFn: () =>
      getPredictions({
        cityId,
        horizonMinutes,
        bbox: bboxParam,
      }),
    refetchInterval: bootstrapQ.data?.config.refresh_seconds.predictions
      ? bootstrapQ.data.config.refresh_seconds.predictions * 1000
      : 60000,
  });

  const isLoading =
    bootstrapQ.isLoading || incidentsQ.isLoading || predictionsQ.isLoading;

  const isError =
    bootstrapQ.isError || incidentsQ.isError || predictionsQ.isError;

  const bootstrapData = bootstrapQ.data || {};
  const lookups = bootstrapQ.data?.lookups || [];

  const incidentsData = incidentsQ.data?.incidents ?? [];
  const predictionsData = predictionsQ.data?.predictions ?? [];

  const cityName = getLookupValue(lookups?.cities, cityId);

  return {
    bboxParam,
    bootstrapData,
    lookups,
    cityName,
    incidentsData,
    predictionsData,
    isLoading,
    isError,
  };
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function usePredictionDetails(predictionId: string | null) {
  return useQuery({
    queryKey: ["prediction-details", predictionId],
    enabled: Boolean(predictionId),
    queryFn: async () => {
      await sleep(500);
      return getPredictionDetails(predictionId!);
    },
    staleTime: 60_000, // cache for 1 min
  });
}

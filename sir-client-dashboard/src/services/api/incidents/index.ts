import { apiClient } from "../../apiClient";
import { ApiClientSchema } from "../../apiClient.types";
import { IncidentsResponseSchema } from "./types";

const Schema = ApiClientSchema(IncidentsResponseSchema);

export async function getIncidents(params: { cityId: string; bbox?: string }) {
  const res = await apiClient.get("/api/v1/incidents", {
    params: {
      city_id: params.cityId,
      bbox: params.bbox,
    },
  });

  const parsed = Schema.parse(res.data);
  return parsed.data ?? { incidents: [], window: { from: "", to: "" } };
}

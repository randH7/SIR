import { apiClient } from "../../apiClient";
import { ApiClientSchema } from "../../apiClient.types";
import { ConfigSchema, LookupsSchema } from "./types";

const ConfigEnvelope = ApiClientSchema(ConfigSchema);
const LookupsEnvelope = ApiClientSchema(LookupsSchema);

export async function getMapConfig(cityId: string) {
  const res = await apiClient.get("/api/v1/config", { params: { city_id: cityId } });
  const parsed = ConfigEnvelope.parse(res.data);

  if (!parsed.data) throw new Error("Config returned null data");
  return parsed.data;
}

export async function getLookups() {
  const res = await apiClient.get("/api/v1/lookups");
  const parsed = LookupsEnvelope.parse(res.data);

  if (!parsed.data) throw new Error("Lookups returned null data");
  return parsed.data;
}

/**
 * Convenience: load both config + lookups together for app bootstrap.
 */
export async function getBootstrap(cityId: string) {
  const [config, lookups] = await Promise.all([
    getMapConfig(cityId),
    getLookups(),
  ]);
  return { config, lookups };
}
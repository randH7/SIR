import { z } from "zod";

export const ConfigSchema = z.object({
  city: z.object({
    id: z.string(),
    name: z.string(),
    center: z.object({ lat: z.number(), lng: z.number() }),
    bbox: z.object({
      min_lat: z.number(),
      min_lng: z.number(),
      max_lat: z.number(),
      max_lng: z.number(),
    }),
  }),
  refresh_seconds: z.object({
    incidents: z.number(),
    predictions: z.number(),
  }),
});

export const LookupItemSchema = z.object({ id: z.string(), value: z.string() });
const RiskLevelSchema = z.object({
  id: z.string(),
  value: z.string(),
  rank: z.number(),
});

export const LookupsSchema = z.object({
  cities: z.array(LookupItemSchema),
  incident_domains: z.array(LookupItemSchema),
  incident_statuses: z.array(LookupItemSchema),
  source_channels: z.array(LookupItemSchema),
  prediction_types: z.array(LookupItemSchema),
  signal_source_types: z.array(LookupItemSchema),
  risk_levels: z.array(RiskLevelSchema),
});

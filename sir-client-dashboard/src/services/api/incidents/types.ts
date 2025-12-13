import { z } from "zod";

export const IncidentsResponseSchema = z.object({
  incidents: z.array(
    z.object({
      incident_id: z.string(),
      title: z.string(),
      domain: z.string(),
      severity: z.number(),
      status: z.string(),
      occurred_at: z.string(),
      updated_at: z.string(),
      lat: z.number(),
      lng: z.number(),
      source_channels: z.array(z.string()),
    })
  ),
  window: z.object({ from: z.string(), to: z.string() }),
});

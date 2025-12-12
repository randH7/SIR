import { z } from "zod";

export const LatLngSchema = z.object({
  lat: z.number(),
  lng: z.number(),
});

export const IncidentSchema = z.object({
  id: z.string(),
  title: z.string(),
  domain: z.enum(["traffic", "drugs", "health", "crowd", "weapons", "other"]),
  severity: z.number().int().min(1).max(5),
  occurredAt: z.string(), // ISO
  location: LatLngSchema,
});

export const PredictionSchema = z.object({
  id: z.string(),
  theme: z.enum([
    "drifting",
    "illegal_gathering",
    "weapons",
    "crowding",
    "crash_risk",
    "other",
  ]),
  confidence: z.number().min(0).max(1),
  timeWindow: z.object({
    start: z.string(), // ISO
    end: z.string(), // ISO
  }),
  geoArea: z.object({
    center: LatLngSchema,
    radiusMeters: z.number().positive(),
  }),
});

export const RadarDatasetSchema = z.object({
  incidents: z.array(IncidentSchema),
  predictions: z.array(PredictionSchema),
});

export type LatLng = z.infer<typeof LatLngSchema>;
export type Incident = z.infer<typeof IncidentSchema>;
export type Prediction = z.infer<typeof PredictionSchema>;
export type RadarDataset = z.infer<typeof RadarDatasetSchema>;

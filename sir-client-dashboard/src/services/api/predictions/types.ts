import { z } from "zod";

export const PredictionsResponseSchema = z.object({
  predictions: z.array(
    z.object({
      prediction_id: z.string(),
      type: z.string(),
      theme: z.string(),
      confidence: z.number(),
      risk_level: z.string(),
      window_start: z.string(),
      window_end: z.string(),
      center_lat: z.number(),
      center_lng: z.number(),
      radius_meters: z.number(),
      summary: z.string().nullable(),
    })
  ),
  horizon_minutes: z.number(),
});

export const PredictionDetailsSchema = z.object({
  prediction: z.object({
    prediction_id: z.string(),
    type: z.string(),
    theme: z.string(),
    confidence: z.number(),
    risk_level: z.string(),
    created_at: z.string(),
    window_start: z.string(),
    window_end: z.string(),
    center_lat: z.number(),
    center_lng: z.number(),
    radius_meters: z.number(),
    summary: z.string().nullable(),
  }),
  explainability: z.object({
    explanations: z.array(
      z.object({
        signal_type: z.string(),
        weight: z.number(),
        summary: z.string(),
      })
    ),
    signals: z.array(
      z.object({
        signal_id: z.string(),
        source_type: z.string(),
        theme: z.string(),
        confidence: z.number(),
        window_start: z.string(),
        window_end: z.string(),
        center_lat: z.number(),
        center_lng: z.number(),
        radius_meters: z.number(),
        payload: z.unknown(),
        weight: z.number(),
        summary: z.string().nullable(),
      })
    ),
  }),
  related: z.object({
    incidents_count_24h: z.number(),
    top_domains: z.array(z.object({ domain: z.string(), count: z.number() })),
    top_severity: z.array(
      z.object({ severity: z.number(), count: z.number() })
    ),
    sample_incidents: z.array(
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
  }),
});

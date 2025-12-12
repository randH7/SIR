import { RadarDataset, RadarDatasetSchema } from "../types/schemas";

const raw: RadarDataset = {
  incidents: [
    {
      id: "inc-1001",
      title: "Traffic collision reported",
      domain: "traffic",
      severity: 3,
      occurredAt: new Date(Date.now() - 25 * 60 * 1000).toISOString(),
      location: { lat: 24.7136, lng: 46.6753 }, // Riyadh-ish
    },
    {
      id: "inc-1002",
      title: "Overcrowding near event area",
      domain: "crowd",
      severity: 4,
      occurredAt: new Date(Date.now() - 70 * 60 * 1000).toISOString(),
      location: { lat: 24.7202, lng: 46.6641 },
    },
    {
      id: "inc-1003",
      title: "Medical assistance requested",
      domain: "health",
      severity: 2,
      occurredAt: new Date(Date.now() - 110 * 60 * 1000).toISOString(),
      location: { lat: 24.7069, lng: 46.6928 },
    },
  ],
  predictions: [
    {
      id: "pred-2001",
      theme: "crash_risk",
      confidence: 0.76,
      timeWindow: {
        start: new Date(Date.now() + 30 * 60 * 1000).toISOString(),
        end: new Date(Date.now() + 90 * 60 * 1000).toISOString(),
      },
      geoArea: {
        center: { lat: 24.7112, lng: 46.6849 },
        radiusMeters: 900,
      },
    },
    {
      id: "pred-2002",
      theme: "crowding",
      confidence: 0.64,
      timeWindow: {
        start: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
        end: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),
      },
      geoArea: {
        center: { lat: 24.728, lng: 46.669 },
        radiusMeters: 1200,
      },
    },
  ],
};

// validate once at module load (fail fast)
export const dataset = RadarDatasetSchema.parse(raw);

import z from "zod";
import { IncidentsResponseSchema } from "../../../../services/api/incidents/types";

type IncidentsResponse = z.infer<typeof IncidentsResponseSchema>;

export type Incidents = IncidentsResponse["incidents"];

export type IncidentItem = IncidentsResponse["incidents"][number];

export interface IncidentsLayerProps {
  incidents: IncidentItem[];
  onSelect: (incidentId: string) => void;
  selectedId: string | null;
}

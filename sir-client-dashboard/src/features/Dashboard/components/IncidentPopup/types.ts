import { IncidentItem } from "../IncidentsLayer/types";

export interface IncidentPopupProps {
  incident: IncidentItem;
  onClose: () => void;
}

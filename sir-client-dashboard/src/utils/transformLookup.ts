import z from "zod";
import { LookupItemSchema } from "../services/api/bootstrap/types";
export type LookupItem = z.infer<typeof LookupItemSchema>;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const getLookupValue = (items: any, id: string): string => {
  const found = items?.find((x: { id: string; }) => x.id === id);
  return found?.value ?? id;
};

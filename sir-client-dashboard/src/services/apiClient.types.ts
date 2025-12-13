import { z } from "zod";

export const ApiClientSchema = <T extends z.ZodTypeAny>(dataSchema: T) =>
  z.object({
    data: dataSchema.nullable(),
    meta: z.object({
      request_id: z.string(),
      generated_at: z.string(),
    }),
    errors: z.array(
      z.object({
        code: z.string(),
        message: z.string(),
        field: z.string().nullable().optional(),
      })
    ),
  });

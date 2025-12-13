export type EnvelopeMeta = {
  request_id: string;
  generated_at: string; // ISO string
};

export type ErrorItem = {
  code: string;
  message: string;
  field?: string | null;
};

export type Envelope<T> = {
  data: T | null;
  meta: EnvelopeMeta;
  errors: ErrorItem[];
};

export const ok = <T>(data: T, requestId: string): Envelope<T> => ({
  data,
  meta: {
    request_id: requestId,
    generated_at: new Date().toISOString(),
  },
  errors: [],
});

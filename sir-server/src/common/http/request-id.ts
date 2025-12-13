import crypto from 'node:crypto';
import type { Request } from 'express';

export const getRequestId = (req: Request): string => {
  const raw = req.headers['x-request-id'];

  if (typeof raw === 'string' && raw.trim().length > 0) return raw;
  if (
    Array.isArray(raw) &&
    typeof raw[0] === 'string' &&
    raw[0].trim().length > 0
  )
    return raw[0];

  return crypto.randomUUID();
};

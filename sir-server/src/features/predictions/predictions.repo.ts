import { Injectable } from '@nestjs/common';
import { DataSource } from 'typeorm';

type PredictionBaseRow = {
  prediction_id: string;
  type: string;
  theme: string;
  confidence: number;
  risk_level: string;
  created_at: string;
  window_start: string;
  window_end: string;
  center_lat: number;
  center_lng: number;
  radius_meters: number;
  summary: string | null;
};

type PredictionSignalRow = {
  signal_id: string;
  source_type: string;
  theme: string;
  confidence: number;
  window_start: string;
  window_end: string;
  center_lat: number;
  center_lng: number;
  radius_meters: number;
  payload: unknown; // JSONB
  weight: number;
  summary: string | null;
};

type RelatedCountRow = { incidents_count_24h: number };
type TopBucketRow = { key: string; count: number };

type RelatedIncidentRow = {
  incident_id: string;
  title: string;
  domain: string;
  severity: number;
  status: string;
  occurred_at: string;
  updated_at: string;
  lat: number;
  lng: number;
  source_channels: string[];
};

@Injectable()
export class PredictionsRepo {
  constructor(private readonly dataSource: DataSource) {}

  async list(params: {
    cityId: string;
    horizonMinutes: number;
    bbox?: string;
    minConfidence: number;
  }): Promise<PredictionBaseRow[]> {
    const sql = `
      WITH bbox AS (
        SELECT
          CASE WHEN $4::text IS NULL THEN NULL ELSE
            ST_MakeEnvelope(
              split_part($4, ',', 1)::double precision,
              split_part($4, ',', 2)::double precision,
              split_part($4, ',', 3)::double precision,
              split_part($4, ',', 4)::double precision,
              4326
            )::geography
          END AS g
      )
      SELECT
        p.prediction_id,
        p.type_id AS type,
        p.theme,
        p.confidence,
        p.risk_level_id AS risk_level,
        p.window_start,
        p.window_end,
        ST_Y(p.center::geometry) AS center_lat,
        ST_X(p.center::geometry) AS center_lng,
        p.radius_meters,
        p.summary
      FROM predictions p
      CROSS JOIN bbox b
      WHERE
        p.city_id = $1
        AND p.window_end >= now()
        AND p.window_start <= now() + make_interval(mins => $2::int)
        AND p.confidence >= $3
        AND (b.g IS NULL OR ST_Intersects(p.center, b.g))
      ORDER BY p.confidence DESC, p.window_start ASC;
    `;

    const rows = await this.dataSource.query<PredictionBaseRow[]>(sql, [
      params.cityId,
      params.horizonMinutes,
      params.minConfidence,
      params.bbox ?? null,
    ]);

    return rows;
  }

  async getBase(predictionId: string): Promise<PredictionBaseRow | null> {
    const sql = `
      SELECT
        p.prediction_id,
        p.type_id AS type,
        p.theme,
        p.confidence,
        p.risk_level_id AS risk_level,
        p.created_at,
        p.window_start,
        p.window_end,
        ST_Y(p.center::geometry) AS center_lat,
        ST_X(p.center::geometry) AS center_lng,
        p.radius_meters,
        p.summary
      FROM predictions p
      WHERE p.prediction_id = $1
      LIMIT 1;
    `;
    const rows = await this.dataSource.query<PredictionBaseRow[]>(sql, [
      predictionId,
    ]);
    return rows[0] ?? null;
  }

  async getSignals(predictionId: string): Promise<PredictionSignalRow[]> {
    const sql = `
      SELECT
        s.signal_id,
        s.source_type_id AS source_type,
        s.theme,
        s.confidence,
        s.window_start,
        s.window_end,
        ST_Y(s.center::geometry) AS center_lat,
        ST_X(s.center::geometry) AS center_lng,
        s.radius_meters,
        s.payload,
        ps.weight,
        ps.summary
      FROM prediction_signals ps
      JOIN signals s ON s.signal_id = ps.signal_id
      WHERE ps.prediction_id = $1
      ORDER BY ps.weight DESC;
    `;
    return this.dataSource.query<PredictionSignalRow[]>(sql, [predictionId]);
  }

  async getRelatedCount(predictionId: string): Promise<number> {
    const sql = `
      SELECT COUNT(*)::int AS incidents_count_24h
      FROM incidents i
      JOIN predictions p ON p.prediction_id = $1
      WHERE
        i.city_id = p.city_id
        AND i.occurred_at >= now() - interval '24 hours'
        AND ST_DWithin(i.location, p.center, p.radius_meters);
    `;
    const rows = await this.dataSource.query<RelatedCountRow[]>(sql, [
      predictionId,
    ]);
    return rows[0]?.incidents_count_24h ?? 0;
  }

  async getTopDomains(
    predictionId: string,
  ): Promise<Array<{ domain: string; count: number }>> {
    const sql = `
      SELECT i.domain_id AS key, COUNT(*)::int AS count
      FROM incidents i
      JOIN predictions p ON p.prediction_id = $1
      WHERE
        i.occurred_at >= now() - interval '24 hours'
        AND ST_DWithin(i.location, p.center, p.radius_meters)
      GROUP BY i.domain_id
      ORDER BY count DESC;
    `;
    const rows = await this.dataSource.query<TopBucketRow[]>(sql, [
      predictionId,
    ]);
    return rows.map((r) => ({ domain: r.key, count: r.count }));
  }

  async getTopSeverity(
    predictionId: string,
  ): Promise<Array<{ severity: number; count: number }>> {
    const sql = `
      SELECT i.severity::text AS key, COUNT(*)::int AS count
      FROM incidents i
      JOIN predictions p ON p.prediction_id = $1
      WHERE
        i.occurred_at >= now() - interval '24 hours'
        AND ST_DWithin(i.location, p.center, p.radius_meters)
      GROUP BY i.severity
      ORDER BY i.severity DESC;
    `;
    const rows = await this.dataSource.query<TopBucketRow[]>(sql, [
      predictionId,
    ]);
    return rows.map((r) => ({ severity: Number(r.key), count: r.count }));
  }

  async getSampleIncidents(
    predictionId: string,
    limit = 10,
  ): Promise<RelatedIncidentRow[]> {
    const sql = `
      SELECT
        i.incident_id,
        i.title,
        i.domain_id AS domain,
        i.severity,
        i.status_id AS status,
        i.occurred_at,
        i.updated_at,
        ST_Y(i.location::geometry) AS lat,
        ST_X(i.location::geometry) AS lng,
        COALESCE(
          array_agg(isc.source_channel_id) FILTER (WHERE isc.source_channel_id IS NOT NULL),
          '{}'
        ) AS source_channels
      FROM incidents i
      LEFT JOIN incident_source_channels isc ON isc.incident_id = i.incident_id
      JOIN predictions p ON p.prediction_id = $1
      WHERE
        i.occurred_at >= now() - interval '24 hours'
        AND ST_DWithin(i.location, p.center, p.radius_meters)
      GROUP BY i.incident_id
      ORDER BY i.occurred_at DESC
      LIMIT $2;
    `;
    return this.dataSource.query<RelatedIncidentRow[]>(sql, [
      predictionId,
      limit,
    ]);
  }
}

import { Injectable } from '@nestjs/common';
import { DataSource } from 'typeorm';

type IncidentRow = {
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
export class IncidentsRepo {
  constructor(private readonly dataSource: DataSource) {}

  async getLast24h(params: {
    cityId: string;
    bbox?: string;
    limit: number;
  }): Promise<IncidentRow[]> {
    const sql = `
      WITH bbox AS (
        SELECT
          CASE WHEN $2::text IS NULL THEN NULL ELSE
            ST_MakeEnvelope(
              split_part($2, ',', 1)::double precision,
              split_part($2, ',', 2)::double precision,
              split_part($2, ',', 3)::double precision,
              split_part($2, ',', 4)::double precision,
              4326
            )::geography
          END AS g
      )
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
      CROSS JOIN bbox b
      WHERE
        i.city_id = $1
        AND i.occurred_at >= now() - interval '24 hours'
        AND (b.g IS NULL OR ST_Intersects(i.location, b.g))
      GROUP BY i.incident_id
      ORDER BY i.occurred_at DESC
      LIMIT $3;
    `;

    const rows = await this.dataSource.query<IncidentRow[]>(sql, [
      params.cityId,
      params.bbox ?? null,
      params.limit,
    ]);

    return rows;
  }
}

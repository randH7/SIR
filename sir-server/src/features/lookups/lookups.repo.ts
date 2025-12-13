import { Injectable } from '@nestjs/common';
import { DataSource } from 'typeorm';

type LookupRow = { id: string; value: string };
type RiskRow = { id: string; value: string; rank: number };

@Injectable()
export class LookupsRepo {
  constructor(private readonly dataSource: DataSource) {}

  private async list(sql: string): Promise<LookupRow[]> {
    return this.dataSource.query<LookupRow[]>(sql);
  }

  async getAll() {
    const [
      cities,
      incident_domains,
      incident_statuses,
      source_channels,
      prediction_types,
      signal_source_types,
      risk_levels,
    ] = await Promise.all([
      this.list(`SELECT id, value FROM cities ORDER BY value ASC;`),
      this.list(`SELECT id, value FROM incident_domains ORDER BY value ASC;`),
      this.list(`SELECT id, value FROM incident_statuses ORDER BY value ASC;`),
      this.list(`SELECT id, value FROM source_channels ORDER BY value ASC;`),
      this.list(`SELECT id, value FROM prediction_types ORDER BY value ASC;`),
      this.list(
        `SELECT id, value FROM signal_source_types ORDER BY value ASC;`,
      ),
      this.dataSource.query<RiskRow[]>(
        `SELECT id, value, rank FROM risk_levels ORDER BY rank ASC;`,
      ),
    ]);

    return {
      cities,
      incident_domains,
      incident_statuses,
      source_channels,
      prediction_types,
      signal_source_types,
      risk_levels,
    };
  }
}

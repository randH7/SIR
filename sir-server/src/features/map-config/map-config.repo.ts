import { Injectable } from '@nestjs/common';
import { DataSource } from 'typeorm';

type CityRow = { id: string; value: string };

@Injectable()
export class MapConfigRepo {
  constructor(private readonly dataSource: DataSource) {}

  async getCity(cityId: string): Promise<CityRow | null> {
    const rows = await this.dataSource.query<CityRow[]>(
      `SELECT id, value FROM cities WHERE id = $1 LIMIT 1;`,
      [cityId],
    );

    return rows[0] ?? null;
  }
}

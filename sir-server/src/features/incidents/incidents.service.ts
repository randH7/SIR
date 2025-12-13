import { Injectable } from '@nestjs/common';
import { IncidentsRepo } from './incidents.repo';

@Injectable()
export class IncidentsService {
  constructor(private readonly repo: IncidentsRepo) {}

  async getIncidentsLast24h(input: {
    cityId?: string;
    bbox?: string;
    limit?: number;
  }) {
    const cityId = input.cityId ?? 'riyadh';
    const limit = input.limit ?? 1000;

    const incidents = await this.repo.getLast24h({
      cityId,
      bbox: input.bbox,
      limit,
    });

    return {
      incidents,
      window: {
        from: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        to: new Date().toISOString(),
      },
    };
  }
}

import { Injectable, NotFoundException } from '@nestjs/common';
import { MapConfigRepo } from './map-config.repo';

type CityCenter = { lat: number; lng: number };
type CityBBox = {
  min_lat: number;
  min_lng: number;
  max_lat: number;
  max_lng: number;
};

@Injectable()
export class MapConfigService {
  constructor(private readonly repo: MapConfigRepo) {}

  async getConfig(cityId?: string) {
    const id = cityId ?? 'riyadh';

    const city = await this.repo.getCity(id);
    if (!city) throw new NotFoundException(`City not found: ${id}`);

    // MVP: Static config for Riyadh (can move to DB later)
    const center: CityCenter = { lat: 24.7136, lng: 46.6753 };
    const bbox: CityBBox = {
      min_lat: 24.3,
      min_lng: 46.2,
      max_lat: 25.1,
      max_lng: 47.2,
    };

    return {
      city: {
        id: city.id,
        name: city.value,
        center,
        bbox,
      },
      refresh_seconds: {
        incidents: 15,
        predictions: 60,
      },
    };
  }
}

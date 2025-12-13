import { Module } from '@nestjs/common';
import { MapConfigController } from './map-config.controller';
import { MapConfigService } from './map-config.service';
import { MapConfigRepo } from './map-config.repo';

@Module({
  controllers: [MapConfigController],
  providers: [MapConfigService, MapConfigRepo],
})
export class MapConfigModule {}

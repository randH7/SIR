import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';

import { DatabaseModule } from './database/database.module';
import { MapConfigModule } from './features/map-config/map-config.module';
import { IncidentsModule } from './features/incidents/incidents.module';
import { PredictionsModule } from './features/predictions/predictions.module';
import { LookupsModule } from './features/lookups/lookups.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    DatabaseModule,
    MapConfigModule,
    IncidentsModule,
    PredictionsModule,
    LookupsModule,
  ],
})
export class AppModule {}

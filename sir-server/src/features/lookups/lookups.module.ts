import { Module } from '@nestjs/common';
import { LookupsController } from './lookups.controller';
import { LookupsService } from './lookups.service';
import { LookupsRepo } from './lookups.repo';

@Module({
  controllers: [LookupsController],
  providers: [LookupsService, LookupsRepo],
})
export class LookupsModule {}

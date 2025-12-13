import { Module } from '@nestjs/common';
import { IncidentsController } from './incidents.controller';
import { IncidentsService } from './incidents.service';
import { IncidentsRepo } from './incidents.repo';

@Module({
  controllers: [IncidentsController],
  providers: [IncidentsService, IncidentsRepo],
})
export class IncidentsModule {}

import { Module } from '@nestjs/common';
import { PredictionsController } from './predictions.controller';
import { PredictionsService } from './predictions.service';
import { PredictionsRepo } from './predictions.repo';

@Module({
  controllers: [PredictionsController],
  providers: [PredictionsService, PredictionsRepo],
})
export class PredictionsModule {}

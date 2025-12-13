import { Controller, Get, Param, Query, Req } from '@nestjs/common';
import type { Request } from 'express';

import { ok } from '../../common/http/response.envelope';
import { getRequestId } from '../../common/http/request-id';

import { GetPredictionsDto } from './dto/get-predictions.dto';
import { PredictionsService } from './predictions.service';
import { GetPredictionDetailsParamsDto } from './dto/get-prediction-details.dto';

@Controller('api/v1/predictions')
export class PredictionsController {
  constructor(private readonly service: PredictionsService) {}

  @Get()
  async list(@Query() query: GetPredictionsDto, @Req() req: Request) {
    const requestId = getRequestId(req);

    const data = await this.service.list({
      cityId: query.city_id,
      horizonMinutes: query.horizon_minutes,
      bbox: query.bbox,
      minConfidence: query.min_confidence,
    });

    return ok(data, requestId);
  }

  @Get(':prediction_id')
  async getDetails(
    @Param() params: GetPredictionDetailsParamsDto,
    @Req() req: Request,
  ) {
    const requestId = getRequestId(req);
    const data = await this.service.getDetails(params.prediction_id);
    return ok(data, requestId);
  }
}

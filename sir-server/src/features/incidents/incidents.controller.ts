import { Controller, Get, Query, Req } from '@nestjs/common';
import type { Request } from 'express';

import { ok } from '../../common/http/response.envelope';
import { getRequestId } from '../../common/http/request-id';

import { GetIncidentsDto } from './dto/get-incidents.dto';
import { IncidentsService } from './incidents.service';

@Controller('api/v1/incidents')
export class IncidentsController {
  constructor(private readonly service: IncidentsService) {}

  @Get()
  async getIncidents(@Query() query: GetIncidentsDto, @Req() req: Request) {
    const requestId = getRequestId(req);

    const data = await this.service.getIncidentsLast24h({
      cityId: query.city_id,
      bbox: query.bbox,
      limit: query.limit,
    });

    return ok(data, requestId);
  }
}

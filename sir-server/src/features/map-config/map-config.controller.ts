import { Controller, Get, Query, Req } from '@nestjs/common';
import type { Request } from 'express';

import { ok } from '../../common/http/response.envelope';
import { getRequestId } from '../../common/http/request-id';

import { GetConfigDto } from './dto/get-config.dto';
import { MapConfigService } from './map-config.service';

@Controller('api/v1/config')
export class MapConfigController {
  constructor(private readonly service: MapConfigService) {}

  @Get()
  async getConfig(@Query() query: GetConfigDto, @Req() req: Request) {
    const requestId = getRequestId(req);
    const data = await this.service.getConfig(query.city_id);
    return ok(data, requestId);
  }
}

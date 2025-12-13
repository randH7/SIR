import { Controller, Get, Req } from '@nestjs/common';
import type { Request } from 'express';

import { ok } from '../../common/http/response.envelope';
import { getRequestId } from '../../common/http/request-id';
import { LookupsService } from './lookups.service';

@Controller('api/v1/lookups')
export class LookupsController {
  constructor(private readonly service: LookupsService) {}

  @Get()
  async get(@Req() req: Request) {
    const requestId = getRequestId(req);
    const data = await this.service.getLookups();
    return ok(data, requestId);
  }
}

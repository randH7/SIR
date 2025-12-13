import { Injectable } from '@nestjs/common';
import { LookupsRepo } from './lookups.repo';

@Injectable()
export class LookupsService {
  constructor(private readonly repo: LookupsRepo) {}

  async getLookups() {
    return this.repo.getAll();
  }
}

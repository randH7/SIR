import { IsOptional, IsString } from 'class-validator';

export class GetConfigDto {
  /**
   * City identifier.
   * MVP default is handled in service.
   */
  @IsOptional()
  @IsString()
  city_id?: string;
}

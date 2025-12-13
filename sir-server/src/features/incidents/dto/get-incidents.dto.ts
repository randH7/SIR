import { IsInt, IsOptional, IsString, Min } from 'class-validator';
import { Transform } from 'class-transformer';

export class GetIncidentsDto {
  @IsOptional()
  @IsString()
  city_id?: string;

  /**
   * Viewport bbox: "minLng,minLat,maxLng,maxLat"
   * Example: "46.2,24.3,47.2,25.1"
   */
  @IsOptional()
  @IsString()
  bbox?: string;

  @IsOptional()
  @Transform(({ value }) => (value === undefined ? undefined : Number(value)))
  @IsInt()
  @Min(1)
  limit?: number;
}

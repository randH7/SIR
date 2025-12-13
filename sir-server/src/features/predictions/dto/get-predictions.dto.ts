import { IsInt, IsOptional, IsString, Max, Min } from 'class-validator';
import { Transform } from 'class-transformer';

export class GetPredictionsDto {
  @IsOptional()
  @IsString()
  city_id?: string;

  @IsOptional()
  @Transform(({ value }) => (value === undefined ? undefined : Number(value)))
  @IsInt()
  @Min(1)
  @Max(24 * 60)
  horizon_minutes?: number;

  /**
   * Viewport bbox: "minLng,minLat,maxLng,maxLat"
   */
  @IsOptional()
  @IsString()
  bbox?: string;

  @IsOptional()
  @Transform(({ value }) => (value === undefined ? undefined : Number(value)))
  @Min(0)
  @Max(1)
  min_confidence?: number;
}

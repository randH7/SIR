import { IsString } from 'class-validator';

export class GetPredictionDetailsParamsDto {
  @IsString()
  prediction_id!: string;
}

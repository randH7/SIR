import { Injectable, NotFoundException } from '@nestjs/common';
import { PredictionsRepo } from './predictions.repo';

@Injectable()
export class PredictionsService {
  constructor(private readonly repo: PredictionsRepo) {}

  async list(input: {
    cityId?: string;
    horizonMinutes?: number;
    bbox?: string;
    minConfidence?: number;
  }) {
    const cityId = input.cityId ?? 'riyadh';
    const horizonMinutes = input.horizonMinutes ?? 120;
    const minConfidence = input.minConfidence ?? 0;

    const predictions = await this.repo.list({
      cityId,
      horizonMinutes,
      bbox: input.bbox,
      minConfidence,
    });

    return {
      predictions,
      horizon_minutes: horizonMinutes,
    };
  }

  async getDetails(predictionId: string) {
    const prediction = await this.repo.getBase(predictionId);
    if (!prediction)
      throw new NotFoundException(`Prediction not found: ${predictionId}`);

    const signals = await this.repo.getSignals(predictionId);

    const explanations = signals.map((s) => ({
      signal_type: s.source_type,
      weight: s.weight,
      summary: s.summary ?? '',
    }));

    const [count24h, topDomains, topSeverity, sampleIncidents] =
      await Promise.all([
        this.repo.getRelatedCount(predictionId),
        this.repo.getTopDomains(predictionId),
        this.repo.getTopSeverity(predictionId),
        this.repo.getSampleIncidents(predictionId, 10),
      ]);

    return {
      prediction,
      explainability: {
        explanations,
        signals,
      },
      related: {
        incidents_count_24h: count24h,
        top_domains: topDomains,
        top_severity: topSeverity,
        sample_incidents: sampleIncidents,
      },
    };
  }
}

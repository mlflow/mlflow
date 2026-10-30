import type { IntlShape } from 'react-intl';
import { convertTraceInfoV3ToRunEvalEntry, getAssessmentInfos } from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import {
  ERROR_KEY,
  getAssessmentAggregateOverallFraction,
} from '@databricks/web-shared/genai-traces-table/utils/AggregationUtils';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { getEvalRunComparisonAssessmentAggregates } from './evalRunComparisonAggregates';

export interface EvalComparisonRunWithTraces {
  runUuid: string;
  traceInfos: ModelTraceInfoV3[];
}

export interface ComparisonScorer {
  id: string;
  name: string;
  unit: 'percentagePoints' | 'raw';
  value: number | undefined;
  sampleCount: number;
}

export interface ComparisonRun {
  runUuid: string;
  runName: string;
  traceCount: number;
  isTruncated: boolean;
  scorers: ComparisonScorer[];
}

export interface ComparisonScorerDelta {
  id: string;
  runUuid: string;
  scorerId: string;
  name: string;
  unit: ComparisonScorer['unit'];
  baselineValue: number;
  value: number;
  delta: number;
  direction: 'improved' | 'regressed' | 'unchanged' | 'neutral';
}

export interface EvalRunComparisonSnapshot {
  baselineRunUuid: string;
  runs: ComparisonRun[];
  deltas: ComparisonScorerDelta[];
  comparisons: { runUuid: string; improved: number; regressed: number; unchanged: number; numericChanges: number }[];
  commonSuccessScorerCount: number;
  leaderRunUuid: string | undefined;
}

export const normalizeComparisonRunUuids = (baselineRunUuid: string, compareRunUuids: string[]): string[] => [
  ...new Set([baselineRunUuid, ...compareRunUuids].filter(Boolean)),
];

export const buildEvalRunComparisonSnapshot = (
  intl: IntlShape,
  runsWithTraces: EvalComparisonRunWithTraces[],
  runNames: ReadonlyMap<string, string>,
  truncatedRunUuids: ReadonlySet<string>,
): EvalRunComparisonSnapshot => {
  const runs = runsWithTraces.map<ComparisonRun>(({ runUuid, traceInfos }) => {
    const entries = traceInfos.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace));
    const evalResults = entries.map((currentRunValue) => ({ currentRunValue }));
    const scorers = getAssessmentInfos(intl, entries, []).flatMap<ComparisonScorer>((info) => {
      const isRate = info.dtype === 'pass-fail' || info.dtype === 'boolean';
      if (!isRate && info.dtype !== 'numeric') {
        return [];
      }
      // Names are the underlying assessment identifiers; display labels are not unique.
      const id = JSON.stringify([info.name, info.isOverall, info.isRetrievalAssessment, isRate ? 'rate' : 'numeric']);
      const aggregates = getEvalRunComparisonAssessmentAggregates(info, evalResults);
      const assessments = entries.flatMap((entry) =>
        info.isOverall ? entry.overallAssessments : (entry.responseAssessmentsByName[info.name] ?? []),
      );
      const sampleCount = isRate
        ? Array.from(aggregates.currentCounts ?? []).reduce(
            (sum, [value, count]) => (value !== undefined && value !== ERROR_KEY ? sum + count : sum),
            0,
          )
        : assessments.filter(
            (assessment) =>
              typeof assessment.numericValue === 'number' &&
              Number.isFinite(assessment.numericValue) &&
              !assessment.errorCode,
          ).length;
      const value =
        sampleCount === 0
          ? undefined
          : isRate
            ? getAssessmentAggregateOverallFraction(info, aggregates.currentCounts) * 100
            : aggregates.currentNumericAverage;
      return [
        {
          id,
          name: info.displayName,
          unit: isRate ? 'percentagePoints' : 'raw',
          value: value !== undefined && Number.isFinite(value) ? value : undefined,
          sampleCount,
        },
      ];
    });
    return {
      runUuid,
      runName: runNames.get(runUuid) || runUuid,
      traceCount: traceInfos.length,
      isTruncated: truncatedRunUuids.has(runUuid),
      scorers,
    };
  });
  const baseline = runs[0];
  const baselineScorers = new Map(baseline?.scorers.map((scorer) => [scorer.id, scorer]));
  const deltas = runs.slice(1).flatMap((run) =>
    run.scorers.flatMap<ComparisonScorerDelta>((scorer) => {
      const baselineValue = baselineScorers.get(scorer.id)?.value;
      if (baselineValue === undefined || scorer.value === undefined) {
        return [];
      }
      const delta = scorer.value - baselineValue;
      return [
        {
          id: JSON.stringify([run.runUuid, scorer.id]),
          runUuid: run.runUuid,
          scorerId: scorer.id,
          name: scorer.name,
          unit: scorer.unit,
          baselineValue,
          value: scorer.value,
          delta,
          direction:
            delta === 0 ? 'unchanged' : scorer.unit === 'raw' ? 'neutral' : delta > 0 ? 'improved' : 'regressed',
        },
      ];
    }),
  );
  const comparisons = runs.slice(1).map((run) => {
    const changes = deltas.filter((delta) => delta.runUuid === run.runUuid);
    return {
      runUuid: run.runUuid,
      improved: changes.filter((delta) => delta.direction === 'improved').length,
      regressed: changes.filter((delta) => delta.direction === 'regressed').length,
      unchanged: changes.filter((delta) => delta.unit === 'percentagePoints' && delta.direction === 'unchanged').length,
      numericChanges: changes.filter((delta) => delta.direction === 'neutral').length,
    };
  });
  const commonScorers =
    baseline?.scorers.filter(
      (scorer) =>
        scorer.unit === 'percentagePoints' &&
        runs.every((run) =>
          run.scorers.some((candidate) => candidate.id === scorer.id && candidate.value !== undefined),
        ),
    ) ?? [];
  const commonScores = runs.map(
    (run) =>
      run.scorers.reduce(
        (sum, scorer) => (commonScorers.some((common) => common.id === scorer.id) ? sum + (scorer.value ?? 0) : sum),
        0,
      ) / commonScorers.length,
  );
  const highestScore = Math.max(...commonScores);
  const leaders = runs.filter((_, index) => Math.abs(commonScores[index] - highestScore) < 1e-9);

  return {
    baselineRunUuid: baseline?.runUuid ?? '',
    runs,
    deltas,
    comparisons,
    commonSuccessScorerCount: commonScorers.length,
    leaderRunUuid:
      runs.length > 1 && runs.every((run) => !run.isTruncated) && commonScorers.length > 0 && leaders.length === 1
        ? leaders[0].runUuid
        : undefined,
  };
};

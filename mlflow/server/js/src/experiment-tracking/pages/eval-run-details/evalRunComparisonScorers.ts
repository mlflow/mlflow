import type { IntlShape } from 'react-intl';
import {
  convertTraceInfoV3ToRunEvalEntry,
  getAssessmentInfos,
  type AssessmentInfo,
} from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from their modules.
import {
  ERROR_KEY,
  getAssessmentAggregateOverallFraction,
} from '@databricks/web-shared/genai-traces-table/utils/AggregationUtils';
import { getDisplayOverallScoreAndChange } from '@databricks/web-shared/genai-traces-table/utils/DisplayUtils';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { getEvalRunComparisonAssessmentAggregates } from './evalRunComparisonAggregates';
import type { ScorerValueCount } from './ScorersGrid';
import type { EvalRunConfigurationTraceData } from './useEvalRunComparisonData';

export interface EvalRunComparisonScorer {
  id: string;
  assessmentInfo: AssessmentInfo;
  dtype: AssessmentInfo['dtype'];
  kind: 'rate' | 'numeric' | 'categorical';
  displayScore: string;
  value: number | undefined;
  valueCounts: ScorerValueCount[];
}

export interface EvalRunComparisonScorerRun {
  runUuid: string;
  runName: string;
  status: EvalRunConfigurationTraceData['status'];
  isTruncated: boolean;
  scorers: EvalRunComparisonScorer[];
}

export interface EvalRunComparisonScorerCell {
  scorer: EvalRunComparisonScorer | undefined;
  baselineValue: number | undefined;
  delta: number | undefined;
  isBest: boolean;
}

export interface EvalRunComparisonScorerRow {
  id: string;
  assessmentInfo: AssessmentInfo;
  cells: EvalRunComparisonScorerCell[];
}

export const buildEvalRunComparisonScorers = (
  intl: IntlShape,
  traceInfos: ModelTraceInfoV3[],
): EvalRunComparisonScorer[] => {
  const entries = traceInfos.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace));
  const evalResults = entries.map((currentRunValue) => ({ currentRunValue }));
  // Classify each run independently: a type change must not turn another run's numeric score into a category.
  return getAssessmentInfos(intl, entries, []).map((assessmentInfo) => {
    const aggregates = getEvalRunComparisonAssessmentAggregates(assessmentInfo, evalResults);
    const { displayScore } = getDisplayOverallScoreAndChange(intl, assessmentInfo, aggregates);
    const valueCounts = Array.from(aggregates.currentCounts ?? [])
      .map(([value, count]) => ({ value, count }))
      .sort((a, b) => b.count - a.count);
    const hasValues = valueCounts.some(({ value, count }) => value !== undefined && value !== ERROR_KEY && count > 0);
    const isRate = assessmentInfo.dtype === 'boolean' || assessmentInfo.dtype === 'pass-fail';
    const value = isRate
      ? hasValues
        ? getAssessmentAggregateOverallFraction(assessmentInfo, aggregates.currentCounts) * 100
        : undefined
      : aggregates.currentNumericAverage;
    const finiteValue = value !== undefined && Number.isFinite(value) ? value : undefined;
    return {
      id: JSON.stringify([assessmentInfo.name, assessmentInfo.isOverall, assessmentInfo.isRetrievalAssessment]),
      assessmentInfo,
      dtype: assessmentInfo.dtype,
      kind: isRate ? 'rate' : assessmentInfo.dtype === 'numeric' ? 'numeric' : 'categorical',
      displayScore: finiteValue !== undefined || (assessmentInfo.dtype === 'string' && hasValues) ? displayScore : '—',
      value: finiteValue,
      valueCounts,
    };
  });
};

export const buildEvalRunComparisonScorerRows = (runs: EvalRunComparisonScorerRun[]): EvalRunComparisonScorerRow[] => {
  const scorersByRun = runs.map((run) => new Map(run.scorers.map((scorer) => [scorer.id, scorer])));
  const rowInfos = new Map<string, AssessmentInfo>();
  for (const run of runs) {
    for (const scorer of run.scorers) rowInfos.set(scorer.id, scorer.assessmentInfo);
  }
  return Array.from(rowInfos, ([id, assessmentInfo]) => {
    const baseline = scorersByRun[0]?.get(id);
    const baselineReady = runs[0]?.status === 'ready' && !runs[0].isTruncated;
    const scores = scorersByRun.map((scorers) => scorers.get(id));
    const canRank =
      runs.length > 1 &&
      runs.every(
        (run, index) =>
          run.status === 'ready' &&
          !run.isTruncated &&
          scores[index]?.kind === 'rate' &&
          scores[index]?.value !== undefined,
      );
    const maximum = canRank ? Math.max(...scores.map((scorer) => scorer?.value ?? -Infinity)) : undefined;
    return {
      id,
      assessmentInfo,
      cells: scores.map((scorer, index) => {
        const canCompare =
          index > 0 &&
          baselineReady &&
          runs[index].status === 'ready' &&
          !runs[index].isTruncated &&
          scorer?.kind === baseline?.kind &&
          scorer?.value !== undefined &&
          baseline?.value !== undefined;
        return {
          scorer,
          baselineValue: canCompare ? baseline?.value : undefined,
          delta:
            canCompare && scorer?.value !== undefined && baseline?.value !== undefined
              ? scorer.value - baseline.value
              : undefined,
          isBest: maximum !== undefined && scorer?.value !== undefined && Math.abs(scorer.value - maximum) < 1e-9,
        };
      }),
    };
  }).sort(
    (a, b) => a.assessmentInfo.displayName.localeCompare(b.assessmentInfo.displayName) || a.id.localeCompare(b.id),
  );
};

import type { IntlShape } from 'react-intl';
import {
  convertTraceInfoV3ToRunEvalEntry,
  getAssessmentInfos,
  type AssessmentInfo,
  type RunEvaluationTracesDataEntry,
} from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from their modules.
import {
  doesAssessmentContainErrors,
  ERROR_KEY,
  getAssessmentAggregateOverallFraction,
} from '@databricks/web-shared/genai-traces-table/utils/AggregationUtils';
import { getEvaluationResultAssessmentValue } from '@databricks/web-shared/genai-traces-table/components/GenAiEvaluationTracesReview.utils';
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

// The shared aggregator only buckets numeric scores; the breakdown tooltip lists each value that was scored.
const getNumericValueCounts = (
  assessmentInfo: AssessmentInfo,
  evalResults: { currentRunValue: RunEvaluationTracesDataEntry }[],
): ScorerValueCount[] => {
  const counts = new Map<number, number>();
  for (const { currentRunValue } of evalResults) {
    const assessments = assessmentInfo.isOverall
      ? currentRunValue.overallAssessments
      : currentRunValue.responseAssessmentsByName[assessmentInfo.name];
    for (const assessment of assessments ?? []) {
      const value = doesAssessmentContainErrors(assessment)
        ? undefined
        : getEvaluationResultAssessmentValue(assessment);
      if (typeof value === 'number' && Number.isFinite(value)) counts.set(value, (counts.get(value) ?? 0) + 1);
    }
  }
  return Array.from(counts, ([value, count]) => ({ value, count })).sort((a, b) => b.count - a.count);
};

export const buildEvalRunComparisonScorers = (
  intl: IntlShape,
  traceInfos: ModelTraceInfoV3[],
): EvalRunComparisonScorer[] => {
  const entries = traceInfos.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace));
  const evalResults = entries.map((currentRunValue) => ({ currentRunValue }));
  // Classify each run independently: a type change must not turn another run's numeric score into a category.
  return getAssessmentInfos(intl, entries, []).map((runAssessmentInfo) => {
    // A session-level scorer is logged on one turn per session, so it's summarized over those turns, as in the
    // run page's column headers; the session's other turns aren't missing a value.
    const summarizedResults = runAssessmentInfo.isSessionLevelAssessment
      ? evalResults.filter(
          ({ currentRunValue }) => (currentRunValue.responseAssessmentsByName[runAssessmentInfo.name]?.length ?? 0) > 0,
        )
      : evalResults;
    const assessmentInfo = runAssessmentInfo.isSessionLevelAssessment
      ? (getAssessmentInfos(
          intl,
          summarizedResults.map(({ currentRunValue }) => currentRunValue),
          [],
        ).find((info) => info.name === runAssessmentInfo.name) ?? runAssessmentInfo)
      : runAssessmentInfo;
    const aggregates = getEvalRunComparisonAssessmentAggregates(assessmentInfo, summarizedResults);
    const valueCounts =
      assessmentInfo.dtype === 'numeric'
        ? getNumericValueCounts(assessmentInfo, summarizedResults)
        : Array.from(aggregates.currentCounts ?? [])
            .map(([value, count]) => ({ value, count }))
            .sort((a, b) => b.count - a.count);
    const presentValues = valueCounts
      .filter(({ value, count }) => value !== undefined && value !== ERROR_KEY && count > 0)
      .map(({ value }) => value);
    const hasValues = presentValues.length > 0;
    // A category count covers the values that were scored, not missing or errored results.
    const { displayScore } = getDisplayOverallScoreAndChange(
      intl,
      assessmentInfo.dtype === 'string' ? { ...assessmentInfo, uniqueValues: new Set(presentValues) } : assessmentInfo,
      aggregates,
    );
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

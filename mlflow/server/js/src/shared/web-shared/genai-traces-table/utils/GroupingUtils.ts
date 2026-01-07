import { sortBy } from 'lodash';

import {
  getTotalTokens,
  SESSION_ID_METADATA_KEY,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';

import type {
  AssessmentValueType,
  GroupedTracesResult,
  RunEvaluationResultAssessment,
  RunEvaluationTracesDataEntry,
  SessionAssessmentAggregate,
  TraceGroupByConfig,
  TraceSessionGroup,
  TraceTableRowData,
} from '../types';
import { getEvaluationResultAssessmentValue } from '../components/GenAiEvaluationTracesReview.utils';

/**
 * Checks if an assessment is a session-level assessment.
 * Session-level assessments have the session ID in their metadata.
 */
export function isSessionLevelAssessment(assessment: RunEvaluationResultAssessment | undefined): boolean {
  if (!assessment?.metadata) return false;
  return SESSION_ID_METADATA_KEY in assessment.metadata;
}

/**
 * Gets the session ID from a trace's metadata.
 */
export function getTraceSessionId(trace: ModelTraceInfoV3): string | undefined {
  return trace.trace_metadata?.[SESSION_ID_METADATA_KEY];
}

/**
 * Checks if any traces have session metadata.
 */
export function hasAnySessionTraces(traces: ModelTraceInfoV3[]): boolean {
  return traces.some((trace) => getTraceSessionId(trace) !== undefined);
}

/**
 * Computes whether a session passed for a pass-fail or boolean assessment.
 * A session passes only if ALL traces in the session pass.
 */
function computeSessionPassStatus(traceValues: Array<{ traceId: string; value: AssessmentValueType }>): boolean | null {
  if (traceValues.length === 0) {
    return null;
  }

  let hasAnyValue = false;
  for (const { value } of traceValues) {
    if (value === undefined || value === null) {
      continue;
    }
    hasAnyValue = true;

    // Check for failure
    if (value === 'no' || value === 'NO' || value === false) {
      return false;
    }
    // Check for non-passing values (anything other than 'yes'/true)
    if (value !== 'yes' && value !== 'YES' && value !== true) {
      return null;
    }
  }

  return hasAnyValue ? true : null;
}

/**
 * Computes the average of numeric values.
 */
function computeNumericAverage(traceValues: Array<{ traceId: string; value: AssessmentValueType }>): number | null {
  const numericValues = traceValues
    .map(({ value }) => value)
    .filter((value): value is number => typeof value === 'number' && !isNaN(value));

  if (numericValues.length === 0) {
    return null;
  }

  const sum = numericValues.reduce((acc, val) => acc + val, 0);
  return sum / numericValues.length;
}

/**
 * Counts the number of passing traces.
 */
function countPassingTraces(traceValues: Array<{ traceId: string; value: AssessmentValueType }>): number {
  return traceValues.filter(({ value }) => value === 'yes' || value === 'YES' || value === true).length;
}

/**
 * Computes aggregated assessment data for a session.
 */
function computeSessionAssessmentAggregate(
  traceValues: Array<{ traceId: string; value: AssessmentValueType }>,
): SessionAssessmentAggregate {
  return {
    sessionPassed: computeSessionPassStatus(traceValues),
    numericAverage: computeNumericAverage(traceValues),
    passCount: countPassingTraces(traceValues),
    totalCount: traceValues.filter(({ value }) => value !== undefined && value !== null).length,
    traceValues,
  };
}

/**
 * Computes all assessment aggregates for a session.
 */
function computeAllSessionAssessmentAggregates(
  evaluationResults: RunEvaluationTracesDataEntry[],
): Map<string, SessionAssessmentAggregate> {
  const aggregates = new Map<string, SessionAssessmentAggregate>();

  // Collect all assessment names
  const assessmentNames = new Set<string>();
  for (const result of evaluationResults) {
    for (const name of Object.keys(result.responseAssessmentsByName ?? {})) {
      assessmentNames.add(name);
    }
    // Also include overall assessments
    for (const assessment of result.overallAssessments ?? []) {
      if (assessment.name) {
        assessmentNames.add(assessment.name);
      }
    }
  }

  // Compute aggregates for each assessment
  for (const assessmentName of assessmentNames) {
    const traceValues: Array<{ traceId: string; value: AssessmentValueType }> = [];

    for (const result of evaluationResults) {
      const traceId = result.requestId;

      // Check response assessments
      const assessments = result.responseAssessmentsByName?.[assessmentName];
      if (assessments?.length) {
        const value = getEvaluationResultAssessmentValue(assessments[0]);
        traceValues.push({ traceId, value: value ?? undefined });
      } else {
        // Check overall assessments
        const overallAssessment = result.overallAssessments?.find((a) => a.name === assessmentName);
        if (overallAssessment) {
          const value = getEvaluationResultAssessmentValue(overallAssessment);
          traceValues.push({ traceId, value: value ?? undefined });
        }
      }
    }

    if (traceValues.length > 0) {
      aggregates.set(assessmentName, computeSessionAssessmentAggregate(traceValues));
    }
  }

  return aggregates;
}

/**
 * Gets the first input value from the earliest trace in a session.
 * Uses evaluation results to get the input data.
 */
export function getSessionFirstInput(evaluationResults: RunEvaluationTracesDataEntry[]): Record<string, unknown> | null {
  if (evaluationResults.length === 0) return null;

  // Sort by request time ascending to get the earliest trace
  const sortedResults = sortBy(evaluationResults, (result) =>
    result.requestTime ? new Date(result.requestTime).getTime() : 0,
  );
  const firstResult = sortedResults[0];

  return firstResult?.inputs ?? null;
}

/**
 * Gets the last output value from the latest trace in a session.
 * Uses evaluation results to get the output data.
 */
export function getSessionLastOutput(evaluationResults: RunEvaluationTracesDataEntry[]): Record<string, unknown> | null {
  if (evaluationResults.length === 0) return null;

  // Sort by request time descending to get the latest trace
  const sortedResults = sortBy(evaluationResults, (result) =>
    result.requestTime ? -new Date(result.requestTime).getTime() : 0,
  );
  const lastResult = sortedResults[0];

  return lastResult?.outputs ?? null;
}

/**
 * Gets the total token count across all traces in a session.
 */
export function getSessionTotalTokens(traces: ModelTraceInfoV3[]): number {
  return traces.reduce((sum, trace) => {
    return sum + (getTotalTokens(trace) ?? 0);
  }, 0);
}

/**
 * Gets the total execution duration across all traces in a session (in milliseconds).
 */
export function getSessionTotalDuration(traces: ModelTraceInfoV3[]): number {
  return traces.reduce((sum, trace) => {
    // execution_duration is stored as seconds (string), convert to ms
    const durationSeconds = parseFloat(trace.execution_duration ?? '0');
    if (isNaN(durationSeconds)) return sum;
    return sum + durationSeconds * 1000;
  }, 0);
}

/**
 * Gets the aggregated state for a session.
 * Returns 'OK' if any trace is OK, 'ERROR' if any is ERROR (and none OK), else 'PENDING'.
 */
export function getSessionAggregatedState(traces: ModelTraceInfoV3[]): string {
  const hasError = traces.some((t) => t.state === 'ERROR');
  const hasOk = traces.some((t) => t.state === 'OK');

  if (hasOk) return 'OK';
  if (hasError) return 'ERROR';
  return 'PENDING';
}

/**
 * Creates a TraceSessionGroup from a list of traces and their evaluation results.
 */
function createSessionGroup(
  sessionId: string,
  traces: ModelTraceInfoV3[],
  traceToEvalResult: Map<string, RunEvaluationTracesDataEntry>,
): TraceSessionGroup {
  // Sort traces by request time
  const sortedTraces = sortBy(traces, (trace) => new Date(trace.request_time).getTime());

  // Get evaluation results for traces in this session
  const evaluationResults: RunEvaluationTracesDataEntry[] = [];
  for (const trace of sortedTraces) {
    const traceId = trace.client_request_id || trace.trace_id;
    const evalResult = traceToEvalResult.get(traceId);
    if (evalResult) {
      evaluationResults.push(evalResult);
    }
  }

  // Compute aggregated assessments
  const aggregatedAssessments = computeAllSessionAssessmentAggregates(evaluationResults);

  // Get session start time
  const sessionStartTime = sortedTraces.length > 0 ? sortedTraces[0].request_time : null;

  return {
    sessionId,
    traces: sortedTraces,
    evaluationResults,
    aggregatedAssessments,
    sessionStartTime,
    traceCount: sortedTraces.length,
  };
}

/**
 * Groups traces by session ID.
 *
 * @param traces - All traces to group
 * @param evaluations - Evaluation results mapped by trace/request ID
 * @param groupByConfig - The grouping configuration (null means no grouping)
 * @returns Grouped traces result with rows and session aggregates
 */
export function groupTracesBySession(
  traces: ModelTraceInfoV3[],
  evaluations: RunEvaluationTracesDataEntry[],
  groupByConfig: TraceGroupByConfig | null,
): GroupedTracesResult {
  // Build a map from request ID to evaluation result
  const traceToEvalResult = new Map<string, RunEvaluationTracesDataEntry>();
  for (const evalResult of evaluations) {
    traceToEvalResult.set(evalResult.requestId, evalResult);
  }

  // If no grouping, return flat list
  if (!groupByConfig || groupByConfig.mode !== 'session') {
    const rows: TraceTableRowData[] = traces.map((trace) => ({
      type: 'trace' as const,
      data: trace,
      evaluationResult: traceToEvalResult.get(trace.client_request_id || trace.trace_id),
    }));

    return {
      rows,
      sessionAggregates: new Map(),
      sessionCount: 0,
      hasSessionTraces: hasAnySessionTraces(traces),
    };
  }

  // Group traces by session ID
  const sessionGroups = new Map<string, ModelTraceInfoV3[]>();
  const ungroupedTraces: ModelTraceInfoV3[] = [];

  for (const trace of traces) {
    const sessionId = getTraceSessionId(trace);
    if (sessionId) {
      const group = sessionGroups.get(sessionId) ?? [];
      group.push(trace);
      sessionGroups.set(sessionId, group);
    } else {
      ungroupedTraces.push(trace);
    }
  }

  // Build rows: session groups first, then ungrouped traces
  const rows: TraceTableRowData[] = [];
  const sessionAggregates = new Map<string, Map<string, SessionAssessmentAggregate>>();

  // Sort sessions by earliest trace time (most recent first)
  const sortedSessionIds = Array.from(sessionGroups.keys()).sort((a, b) => {
    const aTraces = sessionGroups.get(a) ?? [];
    const bTraces = sessionGroups.get(b) ?? [];
    const aTime = aTraces.length > 0 ? new Date(aTraces[0].request_time).getTime() : 0;
    const bTime = bTraces.length > 0 ? new Date(bTraces[0].request_time).getTime() : 0;
    return bTime - aTime; // Most recent first
  });

  for (const sessionId of sortedSessionIds) {
    const sessionTraces = sessionGroups.get(sessionId) ?? [];
    const sessionGroup = createSessionGroup(sessionId, sessionTraces, traceToEvalResult);

    rows.push({
      type: 'sessionGroup',
      data: sessionGroup,
    });

    sessionAggregates.set(sessionId, sessionGroup.aggregatedAssessments);
  }

  // Add ungrouped traces
  for (const trace of ungroupedTraces) {
    rows.push({
      type: 'trace',
      data: trace,
      evaluationResult: traceToEvalResult.get(trace.client_request_id || trace.trace_id),
    });
  }

  return {
    rows,
    sessionAggregates,
    sessionCount: sessionGroups.size,
    hasSessionTraces: sessionGroups.size > 0,
  };
}

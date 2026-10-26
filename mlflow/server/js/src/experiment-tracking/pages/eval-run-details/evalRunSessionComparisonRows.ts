import { SESSION_ID_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { SessionComparisonRow, TracesComparisonRow } from '@databricks/web-shared/traces-table';
import type { EvalComparisonRunWithTraces } from './evalRunComparisonData';
import { buildEvalRunTraceComparisonRows } from './evalRunTraceComparisonRows';

const timestamp = (trace: ModelTraceInfoV3): number => {
  const time = Date.parse(trace.request_time);
  return Number.isFinite(time) ? time : Infinity;
};

// A session matches by scenario only when every turn carries the same goal and persona.
const sessionMatchKey = (sessionId: string, traces: ModelTraceInfoV3[]): string => {
  const goal = traces[0]?.trace_metadata?.['mlflow.simulation.goal'];
  const persona = traces[0]?.trace_metadata?.['mlflow.simulation.persona'];
  const isScenario =
    Boolean(goal && persona) &&
    traces.every(
      (trace) =>
        trace.trace_metadata?.['mlflow.simulation.goal'] === goal &&
        trace.trace_metadata?.['mlflow.simulation.persona'] === persona,
    );
  return JSON.stringify(isScenario ? ['scenario', goal, persona] : ['session', sessionId]);
};

/** Preserve real sessions before aligning repeated scenarios; filtering must happen after this join. */
export const buildEvalRunSessionComparisonRows = (
  runs: EvalComparisonRunWithTraces[],
  runNames: ReadonlyMap<string, string>,
  getRunColor: (runUuid: string) => string,
  recordedInputs?: ReadonlyMap<string, string>,
): TracesComparisonRow[] => {
  type Conversation = { sessionId: string; traces: ModelTraceInfoV3[] };
  const matches = new Map<string, Conversation[][]>();
  const standaloneRuns = runs.map((run, runIndex) => {
    const sessions = new Map<string, ModelTraceInfoV3[]>();
    const standalone: ModelTraceInfoV3[] = [];
    for (const trace of run.traceInfos) {
      const sessionId = trace.trace_metadata?.[SESSION_ID_METADATA_KEY];
      if (!sessionId) {
        standalone.push(trace);
      } else {
        const turns = sessions.get(sessionId) ?? [];
        turns.push(trace);
        sessions.set(sessionId, turns);
      }
    }
    for (const [sessionId, traces] of sessions) {
      traces.sort((a, b) => timestamp(a) - timestamp(b) || a.trace_id.localeCompare(b.trace_id));
      const key = sessionMatchKey(sessionId, traces);
      const byRun = matches.get(key) ?? runs.map(() => []);
      byRun[runIndex].push({ sessionId, traces });
      matches.set(key, byRun);
    }
    return { ...run, traceInfos: standalone };
  });
  const sessionRows: SessionComparisonRow[] = [...matches.entries()].flatMap(([key, byRun]) => {
    byRun.forEach((sessions) =>
      sessions.sort((a, b) => {
        const left = timestamp(a.traces[0]);
        const right = timestamp(b.traces[0]);
        const order = left === right ? 0 : left === Infinity ? 1 : right === Infinity ? -1 : right - left;
        return order || a.sessionId.localeCompare(b.sessionId);
      }),
    );
    return Array.from({ length: Math.max(...byRun.map((sessions) => sessions.length)) }, (_, index) => ({
      kind: 'session',
      id: JSON.stringify(['session', key, byRun.map((sessions) => sessions[index]?.sessionId ?? null)]),
      runs: runs.map((run, runIndex) => ({
        runUuid: run.runUuid,
        label: runNames.get(run.runUuid) ?? run.runUuid,
        color: getRunColor(run.runUuid),
        sessionId: byRun[runIndex][index]?.sessionId,
        traces: byRun[runIndex][index]?.traces ?? [],
      })),
    }));
  });
  return [
    ...sessionRows,
    ...buildEvalRunTraceComparisonRows(standaloneRuns, runNames, getRunColor, '', recordedInputs),
  ];
};

export const getComparisonRunTraces = (run: TracesComparisonRow['runs'][number]): ModelTraceInfoV3[] =>
  'traces' in run ? run.traces : run.trace ? [run.trace] : [];

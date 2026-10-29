import { convertTraceInfoV3ToRunEvalEntry, getTraceInfoInputs } from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { getTraceInfoInputPreview } from '@databricks/web-shared/genai-traces-table/utils/TraceUtils';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { TraceComparisonRow } from '@databricks/web-shared/traces-table';
import type { EvalComparisonRunWithTraces } from './evalRunComparisonData';

export const buildEvalRunTraceComparisonRows = (
  runs: EvalComparisonRunWithTraces[],
  runNames: ReadonlyMap<string, string>,
  getRunColor: (runUuid: string) => string,
  search: string,
  recordedInputs?: ReadonlyMap<string, string>,
): TraceComparisonRow[] => {
  const tracesByInput = new Map<string, ModelTraceInfoV3[][]>();
  runs.forEach((run, runIndex) => {
    run.traceInfos.forEach((trace) => {
      // Previews can collide after truncation; prefer the recorded root-span input.
      const input =
        recordedInputs?.get(trace.trace_id) ??
        trace.request ??
        trace.trace_metadata?.['mlflow.traceInputs'] ??
        getTraceInfoInputs(trace);
      let tracesByRun = tracesByInput.get(input);
      if (!tracesByRun) {
        tracesByRun = runs.map(() => []);
        tracesByInput.set(input, tracesByRun);
      }
      tracesByRun[runIndex].push(trace);
    });
  });

  const entries = Array.from(tracesByInput.values()).flatMap((tracesByRun) => {
    // Stable occurrence order pairs repeated inputs even when search results arrive in a different order.
    tracesByRun.forEach((traces) =>
      traces.sort(
        (a, b) => Date.parse(b.request_time) - Date.parse(a.request_time) || a.trace_id.localeCompare(b.trace_id),
      ),
    );
    return Array.from({ length: Math.max(...tracesByRun.map((traces) => traces.length)) }, (_, occurrence) =>
      tracesByRun.map((traces) => {
        const trace = traces[occurrence];
        return trace ? convertTraceInfoV3ToRunEvalEntry(trace) : undefined;
      }),
    );
  });
  const query = search.toLowerCase();
  return entries.flatMap((values) => {
    // Match after joining: a trace-ID search must retain the other runs' answers to this input.
    if (
      query &&
      !values.some(
        (value) =>
          value &&
          (JSON.stringify(value.inputs).toLowerCase().includes(query) ||
            value.inputsId.toLowerCase() === query ||
            value.evaluationId?.toLowerCase().includes(query) ||
            value.fullTraceId?.toLowerCase().includes(query)),
      )
    ) {
      return [];
    }
    const firstTrace = values.find((value) => value?.traceInfo)?.traceInfo;
    if (!firstTrace) {
      return [];
    }
    return [
      {
        id: JSON.stringify(values.map((value) => value?.evaluationId ?? null)),
        input: getTraceInfoInputPreview(firstTrace),
        runs: runs.map((run, index) => ({
          runUuid: run.runUuid,
          label: runNames.get(run.runUuid) ?? run.runUuid,
          color: getRunColor(run.runUuid),
          trace: values[index]?.traceInfo,
        })),
      },
    ];
  });
};

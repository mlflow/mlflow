import { RunEvaluationTracesDataEntry } from '@mlflow/mlflow/src/shared/web-shared/genai-traces-table/types';
import {
  getTraceInfoInputs,
  getTraceInfoOutputs,
} from '@mlflow/mlflow/src/shared/web-shared/genai-traces-table/utils/TraceUtils';

export const getContentfulColumns = (
  evalRows: RunEvaluationTracesDataEntry[],
): { responseHasContent: boolean; inputHasContent: boolean; tokensHasContent: boolean } => {
  let responseHasContent = false;
  let inputHasContent = false;
  let tokensHasContent = false;

  evalRows.forEach((evalRow) => {
    const traceInfo = evalRow.traceInfo;
    if (!traceInfo) {
      return;
    }
    if (getTraceInfoInputs(traceInfo)) {
      inputHasContent = true;
    }
    if (getTraceInfoOutputs(traceInfo)) {
      responseHasContent = true;
    }
    // TODO: consolidate all mlflow specific tags to consts in ModelTraceExplorer
    if (evalRow.traceInfo?.trace_metadata?.['mlflow.trace.tokenUsage']) {
      tokensHasContent = true;
    }
  });

  return { responseHasContent, inputHasContent, tokensHasContent };
};

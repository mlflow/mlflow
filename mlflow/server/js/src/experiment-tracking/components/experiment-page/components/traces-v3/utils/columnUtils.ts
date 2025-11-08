import type { RunEvaluationTracesDataEntry } from '@databricks/web-shared/genai-traces-table';
import { getTraceInfoInputs, getTraceInfoOutputs } from '@databricks/web-shared/genai-traces-table';

export const checkColumnContents = (
  evalRows: RunEvaluationTracesDataEntry[],
): { responseHasContent: boolean; inputHasContent: boolean; tokensHasContent: boolean } => {
  let responseHasContent = false;
  let inputHasContent = false;
  let tokensHasContent = false;

  for (const evalRow of evalRows) {
    const traceInfo = evalRow.traceInfo;
    if (!traceInfo) {
      continue;
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
  }

  return { responseHasContent, inputHasContent, tokensHasContent };
};

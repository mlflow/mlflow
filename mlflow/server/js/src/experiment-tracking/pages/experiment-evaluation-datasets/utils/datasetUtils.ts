import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import {
  getModelTraceId,
  getModelTraceSpanParentId,
  getParsedExpectationValue,
  isV3ModelTraceInfo,
  isV3ModelTraceSpan,
  isV4ModelTraceSpan,
  tryDeserializeAttribute,
} from '@databricks/web-shared/model-trace-explorer';
import { getSpanAttribute } from '@databricks/web-shared/genai-traces-table';
import { compact, isNil } from 'lodash';

// keep this in sync with EvaluationData._process_trace_records
// from the mlflow python SDK
export const extractDatasetInfoFromTraces = (traces: ModelTrace[]) => {
  const rowData = traces.map((trace) => {
    const rootSpan = trace.data.spans.find((span) => !getModelTraceSpanParentId(span));
    if (isNil(rootSpan)) {
      return null;
    }

    const expectations: Record<string, any> = {};
    if (isV3ModelTraceInfo(trace.info)) {
      for (const assessment of trace.info.assessments ?? []) {
        if (!('expectation' in assessment)) {
          continue;
        }
        expectations[assessment.assessment_name] = getParsedExpectationValue(assessment.expectation);
      }
    }

    return {
      inputs:
        isV3ModelTraceSpan(rootSpan) || isV4ModelTraceSpan(rootSpan)
          ? tryDeserializeAttribute(getSpanAttribute(rootSpan.attributes, 'mlflow.spanInputs') as string)
          : rootSpan.inputs,
      expectations,
      source: {
        source_type: 'TRACE',
        source_data: { trace_id: getModelTraceId(trace) },
      },
    };
  });

  return compact(rowData);
};

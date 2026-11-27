import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import {
  getModelTraceId,
  getModelTraceSpanParentId,
  isV3ModelTraceInfo,
  isV3ModelTraceSpan,
  isV4ModelTraceSpan,
  tryDeserializeAttribute,
} from '@databricks/web-shared/model-trace-explorer';
import { getSpanAttribute } from '@databricks/web-shared/genai-traces-table';
import type { ModelTraceSpan } from '@databricks/web-shared/model-trace-explorer';
import { compact, isEqual, isNil } from 'lodash';

const decodeAttributeValue = (value: unknown) => (typeof value === 'string' ? tryDeserializeAttribute(value) : value);

// Wraps OTel GenAI chat messages as `{ messages: [...] }` so they fit a dataset record, which
// requires dict inputs. The inputs are wrapped only when they are a non-empty list of messages
// (each with a string `role`) that equals the decoded `gen_ai.input.messages` attribute of the
// root span; any other value is returned unchanged. The messages themselves are not modified.
// Keep in sync with `_maybe_wrap_genai_messages` in mlflow/entities/evaluation_dataset.py.
export const maybeWrapGenAiMessages = (rootSpan: ModelTraceSpan, inputs: unknown): unknown => {
  if (
    Array.isArray(inputs) &&
    inputs.length > 0 &&
    inputs.every((m) => typeof m === 'object' && m !== null && typeof m.role === 'string') &&
    isEqual(inputs, decodeAttributeValue(getSpanAttribute(rootSpan.attributes, 'gen_ai.input.messages')))
  ) {
    return { messages: inputs };
  }
  return inputs;
};

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
        // the getAssessmentValue util does not deserialize
        // the expectation value, so we do it manually here
        if ('serialized_value' in assessment.expectation) {
          const value = assessment.expectation.serialized_value.value;
          try {
            expectations[assessment.assessment_name] = JSON.parse(value);
          } catch (e) {
            expectations[assessment.assessment_name] = value;
          }
        } else {
          expectations[assessment.assessment_name] = assessment.expectation.value;
        }
      }
    }

    const rawInputs =
      isV3ModelTraceSpan(rootSpan) || isV4ModelTraceSpan(rootSpan)
        ? tryDeserializeAttribute(getSpanAttribute(rootSpan.attributes, 'mlflow.spanInputs') as string)
        : rootSpan.inputs;

    return {
      inputs: maybeWrapGenAiMessages(rootSpan, rawInputs),
      expectations,
      source: {
        source_type: 'TRACE',
        source_data: { trace_id: getModelTraceId(trace) },
      },
    };
  });

  return compact(rowData);
};

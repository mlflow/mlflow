import { MlflowService } from '../sdk/MlflowService';
import {
  type ModelTrace,
  type ModelTraceSpan,
  type ModelTraceSpanV2,
  type ModelTraceSpanV3,
  type ModelTraceInfoV3,
  type Assessment,
  type ExpectationAssessment,
} from '@databricks/web-shared/model-trace-explorer';

/**
 * Fetches trace information and data for a given trace ID.
 *
 * @param traceId - The ID of the trace to fetch
 * @returns Promise resolving to ModelTrace object or undefined if trace cannot be fetched
 */
export async function getTrace(traceId?: string, traceInfo?: ModelTrace['info']): Promise<ModelTrace | undefined> {
  if (!traceId) {
    return undefined;
  }

  const [traceInfoResponse, traceData] = await Promise.all([
    MlflowService.getExperimentTraceInfoV3(traceId),
    MlflowService.getExperimentTraceData(traceId),
  ]);

  return traceData
    ? {
        info: traceInfoResponse?.trace?.trace_info || {},
        data: traceData,
      }
    : undefined;
}

/**
 * Fetches trace information and data for a given trace ID using the legacy API.
 *
 * @param requestId - The ID of the request to fetch
 * @returns Promise resolving to ModelTrace object or undefined if trace cannot be fetched
 */
export async function getTraceLegacy(requestId?: string): Promise<ModelTrace | undefined> {
  if (!requestId) {
    return undefined;
  }

  const [traceInfo, traceData] = await Promise.all([
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    MlflowService.getExperimentTraceInfo(requestId!).then((response) => response.trace_info || {}),
    // get-trace-artifact is only currently supported in mlflow 2.0 apis
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    MlflowService.getExperimentTraceData(requestId!),
  ]);
  return traceData
    ? {
        info: traceInfo,
        data: traceData,
      }
    : undefined;
}

export function isRootSpan(span: ModelTraceSpan): boolean {
  // Check if it's V3 format (has trace_id and span_id fields)
  if ('trace_id' in span && 'span_id' in span) {
    const v3Span = span as ModelTraceSpanV3;
    return !v3Span.parent_span_id;
  }

  // V2 format (has context field)
  const v2Span = span as ModelTraceSpanV2;
  return !v2Span.parent_span_id && !v2Span.parent_id;
}

export function getRootSpan(trace: ModelTrace): ModelTraceSpan | null {
  const spans = trace.data?.spans || [];
  return spans.find(isRootSpan) || null;
}

/**
 * Converts a value to a string representation
 * If already a string, returns as-is. Otherwise, JSON stringifies it.
 */
function ensureString(value: any): string {
  if (typeof value === 'string') {
    return value;
  }
  return JSON.stringify(value);
}

/**
 * Helper function to extract a field from a span, supporting both V2 and V3 formats.
 * V2 format has direct fields (inputs/outputs), while V3 format uses attributes.
 */
function extractFieldFromSpan(
  rootSpan: ModelTraceSpan,
  v2FieldName: 'inputs' | 'outputs',
  v3AttributeKey: string,
): string | null {
  // V2 format - direct field
  if (v2FieldName in rootSpan) {
    const value = (rootSpan as any)[v2FieldName];
    if (value !== undefined) {
      return ensureString(value);
    }
  }

  // V3 format or V2 with attributes - check attributes
  if (rootSpan.attributes?.[v3AttributeKey] !== undefined) {
    return ensureString(rootSpan.attributes[v3AttributeKey]);
  }

  return null;
}

export function extractInputs(trace: ModelTrace): string | null {
  const rootSpan = getRootSpan(trace);
  if (!rootSpan) return null;

  return extractFieldFromSpan(rootSpan, 'inputs', 'mlflow.spanInputs');
}

export function extractOutputs(trace: ModelTrace): string | null {
  const rootSpan = getRootSpan(trace);
  if (!rootSpan) return null;

  return extractFieldFromSpan(rootSpan, 'outputs', 'mlflow.spanOutputs');
}

function isExpectationAssessment(assessment: Assessment): assessment is ExpectationAssessment {
  return 'expectation' in assessment;
}

function extractExpectationValue(expectation: ExpectationAssessment['expectation']): any {
  if ('value' in expectation) {
    return expectation.value;
  }

  if ('serialized_value' in expectation && expectation.serialized_value) {
    try {
      return JSON.parse(expectation.serialized_value.value);
    } catch {
      return expectation.serialized_value.value;
    }
  }

  return null;
}

export function extractExpectations(trace: ModelTrace): Record<string, any> {
  const result: Record<string, any> = {};

  // Check if info is ModelTraceInfoV3 (has assessments)
  const info = trace.info as ModelTraceInfoV3;
  if (!info.assessments || !Array.isArray(info.assessments)) {
    return result;
  }

  // Filter and process assessments
  info.assessments.forEach((assessment: Assessment) => {
    // Only process expectation assessments (not feedback)
    if (!isExpectationAssessment(assessment)) {
      return;
    }

    // Skip invalid assessments
    if (assessment.valid === false) {
      return;
    }

    // Only include HUMAN source type
    if (assessment.source?.source_type !== 'HUMAN') {
      return;
    }

    // Extract the expectation value
    const value = extractExpectationValue(assessment.expectation);
    if (value !== undefined && value !== null) {
      result[assessment.assessment_name] = value;
    }
  });

  return result;
}

/**
 * Single document retrieved from a retrieval span
 */
export interface RetrievedDocument {
  doc_uri: string;
  content: string;
}

/**
 * Retrieval context from a single retrieval span
 */
export interface RetrievalContext {
  span_id: string;
  span_name: string;
  documents: Array<RetrievedDocument>;
}

/**
 * All retrieval contexts for a trace (chat assessments API format)
 * Contains retrieval contexts from all top-level retrieval spans
 */
export interface TraceRetrievalContexts {
  retrieved_documents: Array<RetrievalContext>;
}

/**
 * Check if a span is a retrieval span
 */
function isRetrievalSpan(span: ModelTraceSpan): boolean {
  // V3 format
  if ('trace_id' in span && 'span_id' in span) {
    const v3Span = span as ModelTraceSpanV3;
    let spanType = v3Span.attributes?.['mlflow.spanType'];

    // Handle JSON-encoded string values (e.g., '"RETRIEVER"' -> 'RETRIEVER')
    if (typeof spanType === 'string' && spanType.startsWith('"') && spanType.endsWith('"')) {
      try {
        spanType = JSON.parse(spanType);
      } catch {
        // If parsing fails, use as-is
      }
    }

    return spanType === 'RETRIEVER';
  }

  // V2 format
  const v2Span = span as ModelTraceSpanV2;
  let spanType = v2Span.span_type;

  // Handle JSON-encoded string values (e.g., '"RETRIEVER"' -> 'RETRIEVER')
  if (typeof spanType === 'string' && spanType.startsWith('"') && spanType.endsWith('"')) {
    try {
      spanType = JSON.parse(spanType);
    } catch {
      // If parsing fails, use as-is
    }
  }

  return spanType === 'RETRIEVER';
}

/**
 * Helper to find a span by its ID
 */
function findSpanById(spans: ModelTraceSpan[], spanId: string): ModelTraceSpan | undefined {
  return spans.find((s) => {
    if ('span_id' in s) {
      return (s as ModelTraceSpanV3).span_id === spanId;
    }
    return (s as ModelTraceSpanV2).context.span_id === spanId;
  });
}

/**
 * Get top-level retrieval spans from a trace
 * Top-level retrieval spans are retrieval spans that are not children of other retrieval spans
 *
 * For example, given the following spans:
 * - Span A (Chain)
 *   - Span B (Retriever)
 *     - Span C (Retriever)
 *   - Span D (Retriever)
 *     - Span E (LLM)
 *       - Span F (Retriever)
 * Span B and Span D are top-level retrieval spans.
 * Span C and Span F are NOT top-level retrieval spans because they are children of other retrieval spans.
 *
 * Based on mlflow/genai/utils/trace_utils.py::_get_top_level_retrieval_spans
 */
function getTopLevelRetrievalSpans(trace: ModelTrace): ModelTraceSpan[] {
  const spans = trace.data?.spans || [];
  if (spans.length === 0) {
    return [];
  }

  // Get all retrieval spans
  const retrievalSpans = spans.filter(isRetrievalSpan);
  if (retrievalSpans.length === 0) {
    return [];
  }

  // Build a map of retrieval span IDs for quick lookup
  const retrievalSpanIds = new Set<string>();
  for (const span of retrievalSpans) {
    const spanId = 'span_id' in span ? (span as ModelTraceSpanV3).span_id : (span as ModelTraceSpanV2).context.span_id;
    if (spanId) {
      retrievalSpanIds.add(spanId);
    }
  }

  // Filter to only top-level retrieval spans
  const topLevelSpans: ModelTraceSpan[] = [];

  for (const span of retrievalSpans) {
    let isTopLevel = true;

    // Get initial parent ID based on span format
    let currentParentId: string | null | undefined;
    if ('trace_id' in span && 'span_id' in span) {
      currentParentId = (span as ModelTraceSpanV3).parent_span_id;
    } else {
      const v2Span = span as ModelTraceSpanV2;
      currentParentId = v2Span.parent_span_id || v2Span.parent_id;
    }

    // Check if any ancestor is a retrieval span
    while (currentParentId) {
      if (retrievalSpanIds.has(currentParentId)) {
        isTopLevel = false;
        break;
      }

      // Find parent span to continue traversing (inlined to avoid closure issues)
      let parentSpan: ModelTraceSpan | undefined;
      for (const s of spans) {
        const sId = 'span_id' in s ? (s as ModelTraceSpanV3).span_id : (s as ModelTraceSpanV2).context.span_id;
        if (sId === currentParentId) {
          parentSpan = s;
          break;
        }
      }

      if (!parentSpan) {
        break;
      }

      // Get next parent ID
      if ('trace_id' in parentSpan && 'span_id' in parentSpan) {
        currentParentId = (parentSpan as ModelTraceSpanV3).parent_span_id;
      } else {
        const v2Parent = parentSpan as ModelTraceSpanV2;
        currentParentId = v2Parent.parent_span_id || v2Parent.parent_id;
      }
    }

    if (isTopLevel) {
      topLevelSpans.push(span);
    }
  }

  return topLevelSpans;
}

/**
 * Extract retrieval contexts from a trace
 * Returns retrieval contexts from all top-level retrieval spans
 */
export function extractRetrievalContext(trace: ModelTrace): TraceRetrievalContexts | null {
  const topLevelRetrievalSpans = getTopLevelRetrievalSpans(trace);
  if (topLevelRetrievalSpans.length === 0) {
    return null;
  }

  // Extract documents from each top-level retrieval span
  const retrievalResults: Array<RetrievalContext> = [];

  for (const retrievalSpan of topLevelRetrievalSpans) {
    let retrievedDocuments: Array<RetrievedDocument> = [];
    let spanId: string = '';
    let spanName: string = '';

    // V3 format
    if ('trace_id' in retrievalSpan && 'span_id' in retrievalSpan) {
      const v3Span = retrievalSpan as ModelTraceSpanV3;
      spanId = v3Span.span_id || '';
      spanName = v3Span.name || spanId;
      let outputs = v3Span.attributes?.['mlflow.spanOutputs'];

      // Handle JSON-encoded outputs (parse if it's a string)
      if (typeof outputs === 'string') {
        try {
          outputs = JSON.parse(outputs);
        } catch {
          // If parsing fails, skip this span
          continue;
        }
      }

      // Extract retrieved documents from outputs
      if (outputs && typeof outputs === 'object' && Array.isArray(outputs)) {
        retrievedDocuments = outputs.map((item: any) => ({
          doc_uri: item.doc_uri || item.uri || item.metadata?.doc_uri || '',
          content: item.content || item.page_content || '',
        }));
      }
    } else {
      // V2 format
      const v2Span = retrievalSpan as ModelTraceSpanV2;
      spanId = v2Span.context?.span_id || '';
      spanName = v2Span.name || spanId;

      // Extract retrieved documents from outputs
      if (v2Span.outputs && Array.isArray(v2Span.outputs)) {
        retrievedDocuments = v2Span.outputs.map((item: any) => ({
          doc_uri: item.doc_uri || item.uri || '',
          content: item.content || item.page_content || '',
        }));
      }
    }

    // Add this span's result
    if (retrievedDocuments.length > 0) {
      retrievalResults.push({ span_id: spanId, span_name: spanName, documents: retrievedDocuments });
    }
  }

  // Return null if we couldn't extract any documents
  if (retrievalResults.length === 0) {
    return null;
  }

  return {
    retrieved_documents: retrievalResults,
  };
}

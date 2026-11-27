import {
  SESSION_ID_METADATA_KEY,
  type FeedbackAssessment,
  type IssueReferenceAssessment,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';

const DEFAULT_TRACE_METADATA = {
  'mlflow.trace.tokenUsage': JSON.stringify({ input_tokens: 10, output_tokens: 5, total_tokens: 15 }),
  'mlflow.trace.cost': JSON.stringify({ input_cost: 0.001, output_cost: 0.002, total_cost: 0.003 }),
};

/** Build a minimal V4 trace info fixture. Fields default to sensible values; override as needed. */
export const makeTrace = (id: string, over: Partial<ModelTraceInfoV3> = {}): ModelTraceInfoV3 => ({
  trace_id: id,
  trace_location: {
    type: 'UC_SCHEMA',
    uc_schema: { catalog_name: 'cat', schema_name: 'sch' },
  } as ModelTraceInfoV3['trace_location'],
  request_preview: `request for ${id}`,
  response_preview: `response for ${id}`,
  request_time: '2025-01-01T00:00:00.000Z',
  execution_duration: '1.2s',
  state: 'OK',
  tags: {},
  trace_metadata: DEFAULT_TRACE_METADATA,
  assessments: [],
  ...over,
});

/**
 * Build a trace carrying a session id (merged into the default metadata so token/cost stay intact).
 * Drives the data-driven "Session column visible when the page has sessions" behavior in tests.
 */
export const makeSessionTrace = (id: string, sessionId = `session-${id}`): ModelTraceInfoV3 =>
  makeTrace(id, { trace_metadata: { ...DEFAULT_TRACE_METADATA, [SESSION_ID_METADATA_KEY]: sessionId } });

/**
 * Build a trace carrying user tags plus an internal `mlflow.*` tag. Drives the tags-column tests: the
 * internal tag must be filtered out of the preview, the user tags shown. Defaults to two user tags so
 * "+N more" and the hover list are exercised. In OSS, also includes the `mlflow.trace.spansLocation`
 * tag so the drawer's `getTrace` takes the tracking-store path.
 */
export const makeTaggedTrace = (
  id: string,
  tags: Record<string, string> = { env: 'prod', team: 'ml' },
): ModelTraceInfoV3 =>
  makeTrace(id, {
    tags: {
      ...tags,
      'mlflow.traceInternal': 'hidden',
      'mlflow.trace.spansLocation': 'TRACKING_STORE',
    },
  });

/** Build `count` sequentially-numbered traces (tr-000, tr-001, …). */
export const makeTraces = (count: number, prefix = 'tr'): ModelTraceInfoV3[] =>
  Array.from({ length: count }, (_, i) => makeTrace(`${prefix}-${String(i).padStart(3, '0')}`));

/** Build a feedback assessment fixture. `over` patches any field (e.g. `create_time`, `valid`). */
export const makeFeedbackAssessment = (
  name: string,
  value: string | number | boolean,
  over: Partial<FeedbackAssessment> = {},
): FeedbackAssessment => ({
  assessment_id: `assess-${name}-${value}`,
  assessment_name: name,
  trace_id: 'tr',
  source: { source_type: 'LLM_JUDGE', source_id: 'databricks' },
  create_time: '2025-01-01T00:00:00.000Z',
  last_update_time: '2025-01-01T00:00:00.000Z',
  feedback: { value },
  ...over,
});

/**
 * Build an issue-reference assessment fixture. `issueName` is the display label; the assessment name
 * (the issue id) defaults to `issue-<issueName>`. `over` patches any field.
 */
export const makeIssueAssessment = (
  issueName: string,
  over: Partial<IssueReferenceAssessment> = {},
): IssueReferenceAssessment => ({
  assessment_id: `assess-issue-${issueName}`,
  assessment_name: `issue-${issueName}`,
  trace_id: 'tr',
  source: { source_type: 'LLM_JUDGE', source_id: 'databricks' },
  create_time: '2025-01-01T00:00:00.000Z',
  last_update_time: '2025-01-01T00:00:00.000Z',
  issue: { issue_name: issueName },
  ...over,
});

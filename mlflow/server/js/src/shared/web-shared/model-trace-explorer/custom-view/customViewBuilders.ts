import type { Assessment, ModelTrace, ModelTraceSpanNode } from '../ModelTrace.types';
import { getTotalTokens, isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';

// Must match the `catalogId` declared in `catalog.json`.
export const CUSTOM_VIEW_CATALOG_ID = 'https://mlflow.org/model-trace-explorer/custom-view/catalog.json';

const formatLatencyMs = (ms: number): string => (ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`);

// The metrics we can derive from `modelTraceInfo` alone. All fields are
// display-ready strings so the custom view can bind them directly.
export type TraceMetrics = {
  status: string;
  latency: string;
  totalTokens: string;
  assessments: string;
};

// Extracts the metrics we can derive from `modelTraceInfo` alone, normalizing
// across the V3 and legacy/notebook trace-info shapes. `assessmentCount` is
// passed in (not read off `info.assessments`) so the counter matches the
// AssessmentBoard exactly: it reflects trace-level + span-level assessments AND
// any in-session edits merged from the trace-cached-actions store, keeping the
// counter live after a feedback submit / thumb click.
export const getMetricsFromTraceInfo = (info: ModelTrace['info'], assessmentCount: number): TraceMetrics => {
  if (isV3ModelTraceInfo(info)) {
    const totalTokens = getTotalTokens(info);
    return {
      status: info.state ?? 'STATE_UNSPECIFIED',
      latency: info.execution_duration ?? 'N/A',
      totalTokens: totalTokens != null ? totalTokens.toLocaleString() : 'N/A',
      assessments: String(assessmentCount),
    };
  }

  // Not V3: the guard above narrowed `info` to the legacy/notebook shapes, which
  // both expose `status` and `execution_time_ms` — no cast needed.
  return {
    status: info.status ?? 'UNKNOWN',
    latency: typeof info.execution_time_ms === 'number' ? formatLatencyMs(info.execution_time_ms) : 'N/A',
    totalTokens: 'N/A',
    assessments: String(assessmentCount),
  };
};

export type AssessmentSentiment = 'positive' | 'negative' | 'neutral' | 'error';

export type AssessmentBoardItem = {
  name: string;
  value?: string;
  rationale?: string;
  source?: string;
  sentiment: AssessmentSentiment;
};

export type AgentAssessment = {
  name: string;
  value: unknown;
  rationale?: string;
  source: string;
  spanId?: string;
  error?: string;
};

export type CustomViewData = {
  metrics: TraceMetrics;
  assessmentItems: AssessmentBoardItem[];
};

// An issue reference stores the issue id in `assessment_name` and the readable
// label in `issue.issue_name`, so display the latter exactly as the issues
// section of the assessments pane does.
const getAssessmentDisplayName = (assessment: Assessment): string =>
  'issue' in assessment && assessment.issue
    ? assessment.issue.issue_name || assessment.assessment_name
    : assessment.assessment_name;

// Extracts the displayable value + error from any assessment variant
// (feedback / expectation), so the agent receives real judge/feedback results.
// Issue references carry no verdict — their name is the whole payload.
const getAssessmentValueAndError = (assessment: Assessment): { value: unknown; error?: string } => {
  if ('feedback' in assessment && assessment.feedback) {
    const err = assessment.feedback.error ?? assessment.error;
    return {
      value: assessment.feedback.value,
      error: err ? (err.error_message ?? err.error_code) : undefined,
    };
  }
  if ('expectation' in assessment && assessment.expectation) {
    const expectation = assessment.expectation;
    if ('value' in expectation) {
      return { value: expectation.value };
    }
    if ('serialized_value' in expectation) {
      return { value: expectation.serialized_value.value };
    }
  }
  return { value: undefined, error: assessment.error?.error_message ?? assessment.error?.error_code };
};

// Gathers the trace's raw assessments (trace-level + span-level), deduped by id
// and skipping invalidated ones. Kept as raw `Assessment`s (not the mapped agent
// shape) so callers can merge in-session edits from the trace-cached-actions
// store — keyed by assessment_id — before mapping, exactly like the assessments
// pane. This is what keeps the custom view's counter/board live after a submit.
export const collectTraceAssessments = (
  info: ModelTrace['info'],
  nodeMap: Record<string, ModelTraceSpanNode>,
): Assessment[] => {
  const uniqueAssessmentsById = new Map<string, Assessment>();
  const add = (assessment: Assessment) => {
    if (assessment.valid === false || uniqueAssessmentsById.has(assessment.assessment_id)) {
      return;
    }
    uniqueAssessmentsById.set(assessment.assessment_id, assessment);
  };

  const traceAssessments = isV3ModelTraceInfo(info) ? (info.assessments ?? []) : [];
  for (const assessment of traceAssessments) {
    add(assessment);
  }
  for (const node of Object.values(nodeMap)) {
    for (const assessment of node.assessments ?? []) {
      add(assessment);
    }
  }
  return Array.from(uniqueAssessmentsById.values());
};

// Maps raw assessments into the flat shape the agent prompt / AssessmentBoard
// use, skipping invalidated ones and deduping by id (the input may carry cached
// additions that overlap the base list).
export const mapToAgentAssessments = (assessments: Assessment[]): AgentAssessment[] => {
  const uniqueAssessmentsById = new Map<string, AgentAssessment>();
  for (const assessment of assessments) {
    if (assessment.valid === false || uniqueAssessmentsById.has(assessment.assessment_id)) {
      continue;
    }
    const { value, error } = getAssessmentValueAndError(assessment);
    uniqueAssessmentsById.set(assessment.assessment_id, {
      name: getAssessmentDisplayName(assessment),
      value,
      rationale: assessment.rationale,
      source: assessment.source?.source_type ?? 'SOURCE_TYPE_UNSPECIFIED',
      spanId: assessment.span_id,
      error,
    });
  }
  return Array.from(uniqueAssessmentsById.values());
};

// Collects the trace's real assessments (trace-level + span-level) into the flat
// shape the agent prompt uses. Convenience wrapper over collect + map for callers
// that don't need to merge in-session edits.
export const getAgentAssessments = (
  info: ModelTrace['info'],
  nodeMap: Record<string, ModelTraceSpanNode>,
): AgentAssessment[] => mapToAgentAssessments(collectTraceAssessments(info, nodeMap));

// Maps an assessment's raw value to a verdict polarity for coloring. Affirmative
// values (yes/true/pass/correct) are positive (green); negatives (no/false/fail)
// are negative (red); an error overrides everything; otherwise neutral.
const POSITIVE_VALUES = new Set(['yes', 'true', 'pass', 'passed', 'correct', 'good', 'success']);
const NEGATIVE_VALUES = new Set(['no', 'false', 'fail', 'failed', 'incorrect', 'bad', 'failure']);

const getAssessmentSentiment = ({ value, error }: AgentAssessment): AssessmentSentiment => {
  if (error) {
    return 'error';
  }
  if (typeof value === 'boolean') {
    return value ? 'positive' : 'negative';
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (POSITIVE_VALUES.has(normalized)) {
      return 'positive';
    }
    if (NEGATIVE_VALUES.has(normalized)) {
      return 'negative';
    }
  }
  return 'neutral';
};

export const getAssessmentBoardItems = (assessments: AgentAssessment[]): AssessmentBoardItem[] =>
  assessments.map((assessment) => {
    const hasError = Boolean(assessment.error);
    const hasValue = assessment.value !== undefined && assessment.value !== null;
    return {
      name: assessment.name,
      value: hasError ? 'Error' : hasValue ? String(assessment.value) : undefined,
      rationale: assessment.rationale ?? (hasError ? assessment.error : undefined),
      source: assessment.source,
      sentiment: getAssessmentSentiment(assessment),
    };
  });

// Returns the span's "real" (non-`mlflow.`-prefixed) attributes, mirroring the
// Details & Timeline Attributes tab.
export const getSpanAttributes = (span?: ModelTraceSpanNode): Record<string, unknown> => {
  if (!span?.attributes) {
    return {};
  }
  return Object.fromEntries(Object.entries(span.attributes).filter(([key]) => !key.startsWith('mlflow.')));
};

// Materializes a `{ "$source": "assessments" }` marker into one `AssessmentCard`
// per assessment for the CURRENT trace, returning the child ids to place into
// the host `AssessmentBoard`'s `children` and the components to append.
//
// The generated ids MUST stay a pure function of `idPrefix` + position: cycling
// traces re-resolves the template onto the SAME surface, and A2UI upserts
// components by id. A nondeterministic suffix (hash/timestamp/random) would mint
// fresh ids every render and strand the previous cards in the surface model,
// since nothing deletes them.
export const buildAssessmentCardComponents = (
  items: AssessmentBoardItem[],
  { idPrefix }: { idPrefix: string },
): { childIds: string[]; components: Record<string, unknown>[] } => {
  const components: Record<string, unknown>[] = [];
  const childIds = items.map((item, index) => {
    const id = `${idPrefix}-card-${index}`;
    components.push({
      id,
      component: 'AssessmentCard',
      name: item.name,
      ...(item.value !== undefined ? { value: item.value } : {}),
      ...(item.rationale !== undefined ? { rationale: item.rationale } : {}),
      ...(item.source !== undefined ? { source: item.source } : {}),
      sentiment: item.sentiment,
    });
    return id;
  });
  return { childIds, components };
};

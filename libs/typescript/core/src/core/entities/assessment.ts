/**
 * TypeScript counterparts of Python `mlflow.entities.assessment` for the
 * V3 assessments REST API. Field names on the wire are proto JSON (snake_case).
 */

export const AssessmentSourceType = {
  SOURCE_TYPE_UNSPECIFIED: 'SOURCE_TYPE_UNSPECIFIED',
  HUMAN: 'HUMAN',
  LLM_JUDGE: 'LLM_JUDGE',
  CODE: 'CODE',
} as const;

export type AssessmentSourceTypeName =
  (typeof AssessmentSourceType)[keyof typeof AssessmentSourceType];

const VALID_SOURCE_TYPES = new Set<string>(Object.values(AssessmentSourceType));

export type FeedbackPrimitive = number | string | boolean;
export type FeedbackValueType =
  | FeedbackPrimitive
  | FeedbackPrimitive[]
  | { [key: string]: FeedbackPrimitive | FeedbackValueType };

/** Matches Python `mlflow.entities.assessment_error` truncation. */
export const STACK_TRACE_TRUNCATION_PREFIX = '[Stack trace is truncated]\n...\n';
export const STACK_TRACE_TRUNCATION_LENGTH = 10000;

export interface AssessmentError {
  errorCode?: string;
  errorMessage?: string;
  stackTrace?: string;
}

export interface AssessmentSource {
  sourceType: AssessmentSourceTypeName;
  sourceId: string;
}

export interface SerializedAssessmentSource {
  source_type: string;
  source_id?: string;
}

export interface SerializedAssessmentError {
  error_code?: string;
  error_message?: string;
  stack_trace?: string;
}

export interface SerializedFeedbackValue {
  value?: unknown;
  error?: SerializedAssessmentError;
}

export interface SerializedAssessment {
  assessment_id?: string;
  assessment_name: string;
  trace_id?: string;
  span_id?: string;
  source?: SerializedAssessmentSource;
  create_time?: string;
  last_update_time?: string;
  feedback?: SerializedFeedbackValue;
  expectation?: unknown;
  issue?: unknown;
  rationale?: string;
  error?: SerializedAssessmentError;
  metadata?: Record<string, string>;
  overrides?: string;
  valid?: boolean;
}

export class Feedback {
  name: string;
  source: AssessmentSource;
  value?: FeedbackValueType;
  error?: AssessmentError;
  traceId?: string;
  spanId?: string;
  rationale?: string;
  metadata?: Record<string, string>;
  assessmentId?: string;
  createTime?: string;
  lastUpdateTime?: string;
  overrides?: string;
  valid?: boolean;

  constructor(params: {
    name?: string;
    source?: AssessmentSource;
    value?: FeedbackValueType;
    error?: AssessmentError | string;
    traceId?: string;
    spanId?: string;
    rationale?: string;
    metadata?: Record<string, string>;
    assessmentId?: string;
    createTime?: string;
    lastUpdateTime?: string;
    overrides?: string;
    valid?: boolean;
  }) {
    this.name = params.name ?? 'feedback';
    this.source = params.source
      ? {
          sourceType: standardizeSourceType(params.source.sourceType),
          sourceId: params.source.sourceId || 'default',
        }
      : { sourceType: AssessmentSourceType.CODE, sourceId: 'default' };
    this.value = params.value;
    this.error =
      typeof params.error === 'string'
        ? { errorCode: 'ASSESSMENT_ERROR', errorMessage: params.error }
        : params.error;
    this.traceId = params.traceId;
    this.spanId = params.spanId;
    this.rationale = params.rationale;
    this.metadata = params.metadata;
    this.assessmentId = params.assessmentId;
    this.createTime = params.createTime;
    this.lastUpdateTime = params.lastUpdateTime;
    this.overrides = params.overrides;
    this.valid = params.valid;
  }

  toJson(): SerializedAssessment {
    const json: SerializedAssessment = {
      assessment_name: this.name,
      source: {
        source_type: this.source.sourceType,
        source_id: this.source.sourceId,
      },
    };
    if (this.assessmentId != null) {
      json.assessment_id = this.assessmentId;
    }
    if (this.traceId != null) {
      json.trace_id = this.traceId;
    }
    if (this.spanId != null) {
      json.span_id = this.spanId;
    }
    if (this.createTime != null) {
      json.create_time = this.createTime;
    }
    if (this.lastUpdateTime != null) {
      json.last_update_time = this.lastUpdateTime;
    }
    if (this.rationale != null) {
      json.rationale = this.rationale;
    }
    if (this.metadata != null) {
      json.metadata = this.metadata;
    }
    if (this.overrides != null) {
      json.overrides = this.overrides;
    }
    if (this.valid != null) {
      json.valid = this.valid;
    }
    json.feedback = {};
    if (this.value !== undefined) {
      json.feedback.value = this.value;
    }
    if (this.error != null) {
      json.feedback.error = assessmentErrorToJson(this.error);
    }
    return json;
  }

  static fromJson(json: SerializedAssessment): Feedback {
    const feedbackValue = json.feedback;
    const topLevelError = json.error;
    const errorJson = feedbackValue?.error ?? topLevelError;
    return new Feedback({
      name: json.assessment_name,
      source: json.source
        ? {
            sourceType: standardizeSourceType(json.source.source_type),
            sourceId: json.source.source_id || 'default',
          }
        : undefined,
      value: feedbackValue?.value as FeedbackValueType | undefined,
      error: errorJson
        ? {
            errorCode: errorJson.error_code,
            errorMessage: errorJson.error_message,
            stackTrace: errorJson.stack_trace,
          }
        : undefined,
      traceId: json.trace_id,
      spanId: json.span_id,
      rationale: json.rationale,
      metadata: json.metadata,
      assessmentId: json.assessment_id,
      createTime: json.create_time,
      lastUpdateTime: json.last_update_time,
      overrides: json.overrides,
      valid: json.valid,
    });
  }
}

export type Assessment = Feedback | SerializedAssessment;

export function isFeedback(assessment: Assessment): assessment is Feedback {
  return assessment instanceof Feedback;
}

/** Truncate stack traces the same way as Python `AssessmentError.to_proto`. */
export function truncateStackTrace(stackTrace: string): string {
  if (stackTrace.length <= STACK_TRACE_TRUNCATION_LENGTH) {
    return stackTrace;
  }
  const truncLen = STACK_TRACE_TRUNCATION_LENGTH - STACK_TRACE_TRUNCATION_PREFIX.length;
  return STACK_TRACE_TRUNCATION_PREFIX + stackTrace.slice(-truncLen);
}

export function assessmentErrorToJson(error: AssessmentError): SerializedAssessmentError {
  const json: SerializedAssessmentError = {};
  if (error.errorCode != null) {
    json.error_code = error.errorCode;
  }
  if (error.errorMessage != null) {
    json.error_message = error.errorMessage;
  }
  if (error.stackTrace != null) {
    json.stack_trace = truncateStackTrace(error.stackTrace);
  }
  return json;
}

export function assessmentFromJson(json: SerializedAssessment): Assessment {
  if (json.feedback) {
    return Feedback.fromJson(json);
  }
  return json;
}

export function assessmentToJson(assessment: Assessment): SerializedAssessment {
  if (assessment instanceof Feedback) {
    return assessment.toJson();
  }
  return assessment;
}

export function standardizeSourceType(sourceType: string): AssessmentSourceTypeName {
  const normalized =
    sourceType.toUpperCase() === 'AI_JUDGE' ? 'LLM_JUDGE' : sourceType.toUpperCase();
  if (!VALID_SOURCE_TYPES.has(normalized)) {
    throw new Error(
      `Invalid assessment source type: ${sourceType}. Valid source types: ${[
        ...VALID_SOURCE_TYPES,
      ].join(', ')}`,
    );
  }
  return normalized as AssessmentSourceTypeName;
}

export function assertMetadataStringMap(
  metadata: Record<string, unknown> | undefined,
): Record<string, string> | undefined {
  if (metadata == null) {
    return undefined;
  }
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (typeof value !== 'string') {
      throw new Error(
        `Assessment metadata values must be strings. Got ${typeof value} for key "${key}".`,
      );
    }
    result[key] = value;
  }
  return result;
}

export function assertFeedbackPayload(params: {
  value?: FeedbackValueType;
  error?: AssessmentError | string;
}): void {
  const hasValue = params.value !== undefined;
  const hasError = params.error != null && params.error !== '';
  if (!hasValue && !hasError) {
    throw new Error('logFeedback requires `value` or `error`.');
  }
}

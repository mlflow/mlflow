/**
 * TypeScript counterparts of Python `mlflow.entities.assessment` for the
 * V3 and Databricks V4 assessments REST APIs. Field names on the wire are proto JSON
 * (snake_case).
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

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };
export type ExpectationValueType = Exclude<JsonValue, null>;

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

export interface SerializedExpectationValue {
  value?: unknown;
  serialized_value?: {
    serialization_format?: string;
    value?: string;
  };
}

export interface SerializedV4TraceLocation {
  type: 'UC_SCHEMA' | 'UC_TABLE_PREFIX';
  uc_schema?: {
    catalog_name: string;
    schema_name: string;
  };
  uc_table_prefix?: {
    catalog_name: string;
    schema_name: string;
    table_prefix: string;
  };
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
  expectation?: SerializedExpectationValue;
  issue?: unknown;
  rationale?: string;
  error?: SerializedAssessmentError;
  metadata?: Record<string, string>;
  overrides?: string;
  valid?: boolean;
  trace_location?: SerializedV4TraceLocation;
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
      traceId: traceIdFromJson(json),
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

const JSON_SERIALIZATION_FORMAT = 'JSON_FORMAT';

export class Expectation {
  name: string;
  source: AssessmentSource;
  value: ExpectationValueType;
  traceId?: string;
  spanId?: string;
  metadata?: Record<string, string>;
  assessmentId?: string;
  createTime?: string;
  lastUpdateTime?: string;
  valid?: boolean;

  constructor(params: {
    name: string;
    value: ExpectationValueType;
    source?: AssessmentSource;
    traceId?: string;
    spanId?: string;
    metadata?: Record<string, string>;
    assessmentId?: string;
    createTime?: string;
    lastUpdateTime?: string;
    valid?: boolean;
  }) {
    if (params.value == null) {
      throw new Error('logExpectation requires `value`.');
    }
    this.name = params.name;
    this.source = params.source
      ? {
          sourceType: standardizeSourceType(params.source.sourceType),
          sourceId: params.source.sourceId || 'default',
        }
      : { sourceType: AssessmentSourceType.HUMAN, sourceId: 'default' };
    this.value = params.value;
    this.traceId = params.traceId;
    this.spanId = params.spanId;
    this.metadata = params.metadata;
    this.assessmentId = params.assessmentId;
    this.createTime = params.createTime;
    this.lastUpdateTime = params.lastUpdateTime;
    this.valid = params.valid;
  }

  toJson(): SerializedAssessment {
    const json: SerializedAssessment = {
      assessment_name: this.name,
      source: {
        source_type: this.source.sourceType,
        source_id: this.source.sourceId,
      },
      expectation: serializeExpectationValue(this.value),
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
    if (this.metadata != null) {
      json.metadata = this.metadata;
    }
    if (this.valid != null) {
      json.valid = this.valid;
    }
    return json;
  }

  static fromJson(json: SerializedAssessment): Expectation {
    if (!json.expectation) {
      throw new Error('Invalid expectation assessment: missing expectation value.');
    }
    return new Expectation({
      name: json.assessment_name,
      source: json.source
        ? {
            sourceType: standardizeSourceType(json.source.source_type),
            sourceId: json.source.source_id || 'default',
          }
        : undefined,
      value: deserializeExpectationValue(json.expectation),
      traceId: traceIdFromJson(json),
      spanId: json.span_id,
      metadata: json.metadata,
      assessmentId: json.assessment_id,
      createTime: json.create_time,
      lastUpdateTime: json.last_update_time,
      valid: json.valid,
    });
  }
}

export type Assessment = Feedback | Expectation | SerializedAssessment;

export function isFeedback(assessment: Assessment): assessment is Feedback {
  return assessment instanceof Feedback;
}

export function isExpectation(assessment: Assessment): assessment is Expectation {
  return assessment instanceof Expectation;
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
  if (json.expectation) {
    return Expectation.fromJson(json);
  }
  return json;
}

export function assessmentToJson(assessment: Assessment): SerializedAssessment {
  if (assessment instanceof Feedback || assessment instanceof Expectation) {
    return assessment.toJson();
  }
  return assessment;
}

export function serializeV4TraceLocation(location: string): SerializedV4TraceLocation {
  const parts = location.split('.');
  if (parts.length === 2 && parts.every(Boolean)) {
    const [catalogName, schemaName] = parts;
    return {
      type: 'UC_SCHEMA',
      uc_schema: { catalog_name: catalogName, schema_name: schemaName },
    };
  }
  if (parts.length === 3 && parts.every(Boolean)) {
    const [catalogName, schemaName, tablePrefix] = parts;
    return {
      type: 'UC_TABLE_PREFIX',
      uc_table_prefix: {
        catalog_name: catalogName,
        schema_name: schemaName,
        table_prefix: tablePrefix,
      },
    };
  }
  throw new Error(
    `Invalid UC location: ${location}. Expected format: <catalog>.<schema>[.<table_prefix>].`,
  );
}

function traceIdFromJson(json: SerializedAssessment): string | undefined {
  if (!json.trace_id || !json.trace_location) {
    return json.trace_id;
  }
  const location = json.trace_location.uc_schema
    ? `${json.trace_location.uc_schema.catalog_name}.${json.trace_location.uc_schema.schema_name}`
    : json.trace_location.uc_table_prefix
      ? `${json.trace_location.uc_table_prefix.catalog_name}.${json.trace_location.uc_table_prefix.schema_name}.${json.trace_location.uc_table_prefix.table_prefix}`
      : undefined;
  return location ? `trace:/${location}/${json.trace_id}` : json.trace_id;
}

function serializeExpectationValue(value: ExpectationValueType): SerializedExpectationValue {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return { value };
  }
  let serialized: string | undefined;
  try {
    serialized = JSON.stringify(value);
  } catch {
    throw new Error('Expectation value must be JSON-serializable.');
  }
  if (serialized === undefined) {
    throw new Error('Expectation value must be JSON-serializable.');
  }
  return {
    serialized_value: {
      serialization_format: JSON_SERIALIZATION_FORMAT,
      value: serialized,
    },
  };
}

function deserializeExpectationValue(value: SerializedExpectationValue): ExpectationValueType {
  if (value.serialized_value) {
    if (value.serialized_value.serialization_format !== JSON_SERIALIZATION_FORMAT) {
      throw new Error(
        `Unknown expectation serialization format: ${value.serialized_value.serialization_format}. ` +
          `Only ${JSON_SERIALIZATION_FORMAT} is supported.`,
      );
    }
    if (value.serialized_value.value == null) {
      throw new Error('Invalid serialized expectation: missing value.');
    }
    return JSON.parse(value.serialized_value.value) as ExpectationValueType;
  }
  if (value.value == null) {
    throw new Error('Invalid expectation assessment: missing value.');
  }
  return value.value as ExpectationValueType;
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

import type { TimelineTreeNode } from './timeline-tree';

export const MLFLOW_TRACE_SCHEMA_VERSION_KEY = 'mlflow.trace_schema.version';

// column name for mlflow trace data in inference tables
export const INFERENCE_TABLE_RESPONSE_COLUMN_KEY = 'response';
export const INFERENCE_TABLE_TRACE_COLUMN_KEY = 'trace';

export type ModelTraceExplorerRenderMode = 'default' | 'json';

export enum ModelSpanType {
  LLM = 'LLM',
  CHAIN = 'CHAIN',
  AGENT = 'AGENT',
  TOOL = 'TOOL',
  FUNCTION = 'FUNCTION',
  CHAT_MODEL = 'CHAT_MODEL',
  RETRIEVER = 'RETRIEVER',
  PARSER = 'PARSER',
  EMBEDDING = 'EMBEDDING',
  RERANKER = 'RERANKER',
  MEMORY = 'MEMORY',
  UNKNOWN = 'UNKNOWN',
}

export enum ModelIconType {
  MODELS = 'models',
  DOCUMENT = 'document',
  CONNECT = 'connect',
  SEARCH = 'search',
  SORT = 'sort',
  UNKNOWN = 'unknown',
  FUNCTION = 'function',
  CODE = 'code',
  NUMBERS = 'numbers',
  WRENCH = 'wrench',
  AGENT = 'agent',
  CHAIN = 'chain',
  USER = 'user',
  SYSTEM = 'system',
  SAVE = 'save',
}

/**
 * Represents a single model trace span.
 * Based on https://github.com/mlflow/mlflow/blob/tracing/mlflow/entities/span.py
 *
 * TODO: clean up all deprecated fields after PrPr customers swap over to
 *       the latest version of mlflow tracing
 */
export type ModelTraceSpanV2 = {
  context: {
    span_id: string;
    trace_id: string;
  };
  name: string;
  /* deprecated, renamed to `parent_id` */
  parent_span_id?: string | null;
  parent_id?: string | null;
  /* deprecated, contained in attributes['mlflow.spanType'] */
  span_type?: ModelSpanType | string;
  /* deprecated, migrated to `status_code` and `status_message` */
  status?: ModelTraceStatus;
  status_code?: string;
  status_message?: string | null;
  start_time: number;
  end_time: number;
  /* deprecated, contained in attributes['mlflow.spanInputs'] */
  inputs?: any;
  /* deprecated, contained in attributes['mlflow.spanOutputs'] */
  outputs?: any;
  attributes?: Record<string, any>;
  events?: ModelTraceEvent[];
  /* metadata for ui usage logging */
  type?: ModelSpanType;
};

export type ModelTraceSpanV3 = {
  trace_id: string;
  span_id: string;
  // can be empty
  trace_state: string;
  // can be empty or null
  parent_span_id: string | null;
  name: string;
  start_time_unix_nano: string;
  end_time_unix_nano: string;
  status: {
    code: ModelSpanStatusCode;
    message?: string;
  };
  attributes: Record<string, any>;
  events?: ModelTraceEvent[];
  /* metadata for ui usage logging */
  type?: ModelSpanType;
};

export type ModelTraceSpan = ModelTraceSpanV2 | ModelTraceSpanV3;

export type ModelTraceEvent = {
  name: string;
  /* deprecated as of v3, migrated to `time_unix_nano` */
  timestamp?: number;
  time_unix_nano?: number;
  attributes?: Record<string, any>;
};

export type ModelTraceData = {
  spans: ModelTraceSpan[];
};

/**
 * Represents a single model trace object.
 * Based on https://github.com/mlflow/mlflow/blob/8e44d102e9568d09d9dc376136d13a5a5d1ab46f/mlflow/tracing/types/model.py#L11
 */
export type ModelTrace = {
  /* deprecated, renamed to `data` */
  trace_data?: ModelTraceData;
  /* deprecated, renamed to `info` */
  trace_info?: ModelTraceInfo;
  data: ModelTraceData;
  info: ModelTraceInfoV3 | ModelTraceInfo | NotebookModelTraceInfo;
};

/**
 * Represents the trace data saved in an inference table.
 * https://github.com/databricks/universe/blob/fb8a572602161aa6387ac32593aa24a91518cc32/rag/serving/python/databricks/rag/unpacking/schemas.py#L133-L141
 */
export type ModelTraceInferenceTableData = {
  app_version_id: string;
  start_timestamp: string;
  end_timestamp: string;
  is_truncated: boolean;
  [MLFLOW_TRACE_SCHEMA_VERSION_KEY]: number;
  spans: (Omit<ModelTraceSpan, 'attributes'> & {
    attributes: string;
  })[];
};

export type ModelTraceInfo = {
  request_id?: string;
  experiment_id?: string;
  timestamp_ms?: number;
  execution_time_ms?: number;
  status?: ModelTraceStatus['description'];
  attributes?: Record<string, any>;
  request_metadata?: { key: string; value: string }[];
  tags?: { key: string; value: string }[];
};

// tags and request_metadata in the notebook view
// (i.e. displayed directly from the python client)
// are stored as an object rather than an array.
export type NotebookModelTraceInfo = Omit<ModelTraceInfo, 'tags' | 'request_metadata'> & {
  tags?: { [key: string]: string };
  request_metadata?: { [key: string]: string };
};

export type ModelTraceLocationMlflowExperiment = {
  type: 'MLFLOW_EXPERIMENT';
  mlflow_experiment: {
    experiment_id: string;
  };
};

export type ModelTraceLocationInferenceTable = {
  type: 'INFERENCE_TABLE';
  inference_table: {
    full_table_name: string;
  };
};

export type ModelTraceLocation = ModelTraceLocationMlflowExperiment | ModelTraceLocationInferenceTable;

export type ModelTraceInfoV3 = {
  trace_id: string;
  client_request_id?: string;
  trace_location: ModelTraceLocation;
  request_preview?: string;
  response_preview?: string;
  // timestamp in a format like "2025-02-19T09:52:23.140Z"
  request_time: string;
  // formatted duration string like "32.4s"
  execution_duration: string;
  state: ModelTraceState;
  trace_metadata?: {
    [key: string]: string;
  };
  assessments: Assessment[];
  tags: {
    [key: string]: string;
  };
};

export type ModelTraceState = 'STATE_UNSPECIFIED' | 'OK' | 'ERROR' | 'IN_PROGRESS';

export type ModelSpanStatusCode = 'STATUS_CODE_UNSET' | 'STATUS_CODE_OK' | 'STATUS_CODE_ERROR';

export type ModelTraceStatusUnset = {
  description: 'UNSET';
  status_code: 0;
};

export type ModelTraceStatusOk = {
  description: 'OK';
  status_code: 1;
};

export type ModelTraceStatusError = {
  description: 'ERROR';
  status_code: 2;
};

export type ModelTraceStatusInProgress = {
  description: 'IN_PROGRESS';
  status_code: 3;
};

export enum ModelTraceSpanType {
  LLM = 'LLM',
  CHAIN = 'CHAIN',
  AGENT = 'AGENT',
  TOOL = 'TOOL',
  CHAT_MODEL = 'CHAT_MODEL',
  RETRIEVER = 'RETRIEVER',
  PARSER = 'PARSER',
  EMBEDDING = 'EMBEDDING',
  RERANKER = 'RERANKER',
  MEMORY = 'MEMORY',
  UNKNOWN = 'UNKNOWN',
}

export type ModelTraceStatus =
  | ModelTraceStatusUnset
  | ModelTraceStatusOk
  | ModelTraceStatusError
  | ModelTraceStatusInProgress;

/**
 * Represents a single node in the model trace tree.
 */
export interface ModelTraceSpanNode extends TimelineTreeNode, Pick<ModelTraceSpan, 'attributes' | 'type' | 'events'> {
  assessments: Assessment[];
  inputs?: any;
  outputs?: any;
  children?: ModelTraceSpanNode[];
  chatMessageFormat?: string;
  chatMessages?: ModelTraceChatMessage[];
  chatTools?: ModelTraceChatTool[];
  parentId?: string | null;
  traceId: string;
}

export type ModelTraceExplorerTab = 'chat' | 'content' | 'attributes' | 'events';

export type SearchMatch = {
  span: ModelTraceSpanNode;
  section: 'inputs' | 'outputs' | 'attributes' | 'events';
  key: string;
  isKeyMatch: boolean;
  matchIndex: number;
};

export type SpanFilterState = {
  // always show parents regardless of filter state
  showParents: boolean;
  // always show exceptions regardless of filter state
  showExceptions: boolean;
  // record of span_type: whether to show it
  spanTypeDisplayState: Record<string, boolean>;
};

export interface RetrieverDocument {
  metadata: {
    doc_uri: string;
    chunk_id: string;
    [key: string]: any;
  };
  page_content: string;
  [key: string]: any;
}

export enum CodeSnippetRenderMode {
  JSON = 'json',
  TEXT = 'text',
  MARKDOWN = 'markdown',
  PYTHON = 'python',
}

type ModelTraceTextContentPart = {
  type: 'text' | 'input_text' | 'output_text';
  text: string;
};

type ModelTraceImageUrl = {
  url: string;
  detail?: 'auto' | 'low' | 'high';
};

type ModelTraceImageContentPart = {
  type: 'image_url';
  image_url: ModelTraceImageUrl;
};

type ModelTraceInputAudio = {
  data: string;
  format: 'wav' | 'mp3';
};

type ModelTraceAudioContentPart = {
  type: 'input_audio';
  input_audio: ModelTraceInputAudio;
};

export type ModelTraceContentParts =
  | ModelTraceTextContentPart
  | ModelTraceImageContentPart
  | ModelTraceAudioContentPart;

export type ModelTraceContentType = string | ModelTraceContentParts[];

// We treat content as string in the tracing UI.
export type ModelTraceChatMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool' | 'function' | 'developer';
  name?: string;
  content?: string | null;
  tool_calls?: ModelTraceToolCall[];
  tool_call_id?: string;
};

// The actual chat message schema of mlflow contains string, null and content part list.
export type RawModelTraceChatMessage = Omit<ModelTraceChatMessage, 'content'> & {
  // there are other types, but we don't support them yet
  type?: 'message' | 'reasoning';
  content?: ModelTraceContentType | null;
};

export type ModelTraceChatToolParamProperty = {
  type?: string;
  description?: string;
  enum?: string[];
};

export type ModelTraceChatTool = {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: {
      properties: {
        [key: string]: ModelTraceChatToolParamProperty;
      };
      required?: string[];
    };
  };
};

export type ModelTraceToolCall = {
  id: string;
  function: {
    arguments: string;
    name: string;
  };
};

// aligned to the OpenAI format
export type ModelTraceChatResponse = {
  choices: {
    message: ModelTraceChatMessage;
  }[];
};

export type ModelTraceChatInput = {
  messages: RawModelTraceChatMessage[];
};

export type AssessmentSourceType = 'SOURCE_TYPE_UNSPECIFIED' | 'HUMAN' | 'LLM_JUDGE' | 'CODE';

export interface AssessmentSource {
  source_type: AssessmentSourceType;
  // Identifier for the source. For example:
  // - For a human source -> user name
  // - For an LLM judge -> the judge source (databricks or custom)
  // - For a code judge -> the function name
  source_id: string;
}

export interface AssessmentError {
  error_code: string;
  error_message?: string;
  stack_trace?: string;
}

export type AssessmentValue = string | number | boolean | null | string[];

export interface Feedback {
  // can be null / undefined if error is present
  value?: AssessmentValue;
  error?: AssessmentError;
}

export interface ExpectationValue {
  value: AssessmentValue;
}

export interface ExpectationSerializedValue {
  serialized_value: {
    value: string;
    serialization_format: string;
  };
}

export type Expectation = ExpectationValue | ExpectationSerializedValue;

// should be aligned with `mlflow/api/proto/service.proto`
export interface AssessmentBase {
  assessment_id: string;
  assessment_name: string;
  trace_id: string;
  source: AssessmentSource;
  span_id?: string;

  // the time fields are in the form of a string timestamp
  // e.g. "2025-04-18T04:01:20.159Z"
  create_time: string;
  last_update_time: string;

  rationale?: string;
  metadata?: Record<string, string>;

  // if false, the assessment is not valid and should not be displayed
  // undefined and true should be considered valid.
  valid?: boolean;

  // the assessment_id of the assessment that this assessment overrides
  overrides?: string;

  // UI only field to store the overridden assessment object for easier display
  overriddenAssessment?: Assessment;
}

export interface FeedbackAssessment extends AssessmentBase {
  feedback: Feedback;
}

export interface ExpectationAssessment extends AssessmentBase {
  expectation: Expectation;
}

export type Assessment = FeedbackAssessment | ExpectationAssessment;

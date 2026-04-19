import type { ModelTrace, ModelTraceData } from '../../model-trace-explorer/ModelTrace.types';

// Mirrors the REST entity here: https://src.dev.databricks.com/databricks-eng/universe@master/-/blob/managed-evals/api/proto/review_app_service.proto?L28
export interface ReviewApp {
  review_app_id?: string;
  experiment_id: string;
}

// Mirrors the REST entity here: https://src.dev.databricks.com/databricks-eng/universe@master/-/blob/managed-evals/api/proto/managed_dataset_service.proto?L39

export interface Dataset {
  dataset_id: string;
  create_time: string;
  created_by?: string;
  name?: string;
}

// Mirrors the proto here: https://src.dev.databricks.com/databricks-eng/universe@master/-/blob/managed-evals/api/proto/review_app_service.proto?L187
export interface LabelingSession {
  labeling_session_id: string;
  mlflow_run_id: string;
  name: string;
  dataset?: string;
}

// REST API representation of DatasetRecord (matches proto structure)
export interface DatasetRecordRest {
  dataset_record_id: string;
  create_time: string;
  created_by?: string;
  last_update_time?: string;
  last_updated_by?: string;
  source?: {
    human?: { user_name: string };
    document?: { doc_uri: string; content: string };
    trace?: { trace_id: string };
  };
  inputs: Array<{
    key: string;
    value: any;
  }>;
  expectations?: { [key: string]: { value: any } };
  tags?: { [key: string]: string };
}

// JavaScript representation of DatasetRecord (with inputs as simple dictionary)
export interface DatasetRecordJs {
  dataset_record_id: string;
  create_time: string;
  created_by?: string;
  last_update_time?: string;
  last_updated_by?: string;
  source?: {
    human?: { user_name: string };
    document?: { doc_uri: string; content: string };
    trace?: { trace_id: string };
  };
  inputs: { [key: string]: any };
  expectations?: { [key: string]: any };
  tags?: { [key: string]: string };
}

// Extended item interface with full structure
export interface LabelingSessionItemFull {
  item_id?: string;
  create_time?: string;
  created_by?: string;
  last_update_time?: string;
  last_updated_by?: string;
  source?: {
    dataset_record?: {
      dataset_id?: string;
      dataset_record_id?: string;
    };
    trace_id?: string;
  };
  state?: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'SKIPPED';
  chat_rounds?: Array<{
    trace_id?: string;
    dataset_record?: {
      dataset_id?: string;
      dataset_record_id?: string;
    };
  }>;
}

export interface GetEvalDatasetConfigurationResponse {
  version: string;
  instance_id: string;
  experiment_ids: string[];
}

// https://src.dev.databricks.com/databricks-eng/universe/-/blob/managed-evals/api/proto/managed_dataset_service.proto?L101
export interface BatchCreateDatasetRecordsRequest {
  dataset_id: string;
  requests: CreateDatasetRecordRequest[];
}

export interface BatchCreateDatasetRecordsResponse {
  dataset_records: DatasetRecord[];
}

// https://src.dev.databricks.com/databricks-eng/universe/-/blob/managed-evals/api/proto/managed_dataset_service.proto?L489:9-489:35
export interface CreateDatasetRecordRequest {
  dataset_id: string;
  dataset_record: DatasetRecord;
  should_sync_to_uc: boolean;
}

// https://src.dev.databricks.com/databricks-eng/universe/-/blob/managed-evals/api/proto/managed_dataset_service.proto?L101
export interface DatasetRecord {
  // https://src.dev.databricks.com/databricks-eng/universe@master/-/blob/rag/rag_studio/python/databricks/rag_eval/datasets/rest_entities.py?L28:7-28:13
  source: {
    trace?: {
      trace_id: string;
    };
  };
  inputs: {
    key: string;
    value: any;
  }[];
  // Optional fields for tags and expectations (fixing ML-57011)
  expectations?: Record<string, any>;
  tags?: Record<string, string>;
}

export interface ListDatasetRecordsResponse {
  dataset_records: DatasetRecordRest[];
  next_page_token?: string;
}

export interface LabelingSessionItem {
  source: {
    trace_id?: string;
  };
}

// https://src.dev.databricks.com/databricks-eng/universe/-/blob/managed-evals/api/proto/review_app_service.proto?L746
export interface BatchCreateLabelingSessionItemsRequest {
  review_app_id: string;
  labeling_session_id: string;
  items: LabelingSessionItem[];
}

export interface BatchCreateLabelingSessionItemsResponse {
  items: LabelingSessionItem[];
}

export interface ListLabelingSessionItemsResponse {
  items?: LabelingSessionItemFull[];
  next_page_token?: string;
}

/**
 * Interfaces for the REST API to upsert traces.
 */
export type Trace = {
  trace_id: string;
  trace_info: ModelTrace['info'];
  trace_data: ModelTraceData;
};

export type CreateTracePayload = Omit<Trace, 'trace_id'>;

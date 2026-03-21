export type EvaluationDataset = {
  // Unique identifier for the dataset
  dataset_id: string;

  // Dataset name (user-friendly identifier)
  name: string;

  // Tags as JSON string (key-value pairs for metadata)
  tags?: string;

  // Schema information (JSON)
  schema?: string;

  // Profile information (JSON)
  profile?: string;

  // Dataset digest for integrity checking
  digest?: string;

  // Creation timestamp in milliseconds
  created_time: number;

  // Last update timestamp in milliseconds
  last_update_time: number;

  // User who created the dataset
  created_by: string;

  // User who last updated the dataset
  last_updated_by: string;

  // Associated experiment IDs (populated from entity_associations table)
  experiment_ids: string[];
};

export interface EvaluationDatasetRecord {
  // Unique identifier for the record
  dataset_record_id: string;

  // ID of the dataset this record belongs to
  dataset_id: string;

  // Inputs as JSON string
  inputs?: string;

  // Expectations as JSON string
  expectations?: string;

  // Tags as JSON string
  tags?: string;

  // Source information as JSON string
  source?: string;

  // Source ID for quick lookups (e.g., trace_id)
  source_id?: string;

  // Source type
  source_type?: DatasetRecordSource;

  // Creation timestamp in milliseconds
  created_time?: number;

  // Last update timestamp in milliseconds
  last_update_time?: number;

  // User who created the record
  created_by?: string;

  // User who last updated the record
  last_updated_by?: string;

  // Outputs as JSON string
  outputs?: string;
}

export type SourceType = 'SOURCE_TYPE_UNSPECIFIED' | 'TRACE' | 'HUMAN' | 'DOCUMENT' | 'CODE';

export type DatasetRecordSource = {
  // The type of the source.
  source_type?: SourceType;

  // Source-specific data as JSON
  source_data?: string;
};

export type GetDatasetRecords = {
  // Dataset ID to get records for
  dataset_id: string;

  // Optional pagination - maximum number of records to return
  max_results?: number;

  // Optional pagination token for getting next page
  page_token?: string;
};

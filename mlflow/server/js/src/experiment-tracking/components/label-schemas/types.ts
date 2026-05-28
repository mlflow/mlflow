/**
 * Wire-format types for OSS-native label schemas.
 *
 * These mirror the proto definitions in `mlflow/protos/label_schemas.proto`
 * (snake_case on the wire). Server-side validation rules live in
 * `mlflow/genai/label_schemas/validation.py`; field shapes here are the
 * authoritative client-side contract.
 */

// Mirrors the proto3 enum names in `label_schemas.proto`. The wire
// format is the uppercase enum NAME (proto3 JSON convention); sending
// lowercase silently maps to UNSPECIFIED and the server rejects it.
export type LabelSchemaType = 'FEEDBACK' | 'EXPECTATION';

// Mirrors the proto3 enum names in `label_schemas.proto`. The wire
// format is the uppercase enum NAME (proto3 JSON convention); sending
// lowercase silently maps to UNSPECIFIED and the server drops the
// field. Same gotcha as `LabelSchemaType`.
export type CategoricalSemanticPolarity = 'ASCENDING' | 'DESCENDING';

export interface InputPassFail {
  positive_label: string;
  negative_label: string;
}

export interface InputCategorical {
  options: string[];
  semantic_polarity?: CategoricalSemanticPolarity;
  multi_select?: boolean;
}

export interface InputNumeric {
  min_value?: number;
  max_value?: number;
}

/**
 * Discriminated wrapper matching the proto `LabelSchemaInput` oneof.
 * Exactly one of `pass_fail`, `categorical`, or `numeric` is set on a
 * valid schema; the OSS server rejects an empty wrapper.
 */
export interface LabelSchemaInput {
  pass_fail?: InputPassFail;
  categorical?: InputCategorical;
  numeric?: InputNumeric;
}

export interface LabelSchema {
  schema_id: string;
  experiment_id: string;
  name: string;
  type: LabelSchemaType;
  title: string;
  instruction?: string;
  enable_comment?: boolean;
  input: LabelSchemaInput;
  created_by?: string;
  created_at?: number;
  last_updated_at?: number;
}

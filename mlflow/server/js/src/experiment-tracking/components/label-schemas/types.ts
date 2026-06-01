/**
 * Wire-format types for OSS-native label schemas.
 *
 * These mirror the proto definitions in `mlflow/protos/label_schemas.proto`
 * (snake_case on the wire). Server-side validation rules live in
 * `mlflow/genai/label_schemas/validation.py`; field shapes here are the
 * authoritative client-side contract.
 */

// Mirrors the proto2 enum names in `label_schemas.proto`. The wire
// format is the uppercase enum NAME (proto JSON convention); sending
// lowercase silently maps to UNSPECIFIED and the server rejects it.
export type LabelSchemaType = 'FEEDBACK' | 'EXPECTATION';

export interface InputPassFail {
  positive_label: string;
  negative_label: string;
}

export interface InputCategorical {
  options: string[];
  multi_select?: boolean;
}

export interface InputNumeric {
  min_value?: number;
  max_value?: number;
}

export interface InputText {
  max_length?: number;
}

/**
 * Discriminated wrapper matching the proto `LabelSchemaInput` oneof.
 * Exactly one of `pass_fail`, `categorical`, `numeric`, or `text` is set
 * on a valid schema; the OSS server rejects an empty wrapper.
 */
export interface LabelSchemaInput {
  pass_fail?: InputPassFail;
  categorical?: InputCategorical;
  numeric?: InputNumeric;
  text?: InputText;
}

export interface LabelSchema {
  schema_id: string;
  experiment_id: string;
  name: string;
  type: LabelSchemaType;
  instruction?: string;
  enable_comment?: boolean;
  input: LabelSchemaInput;
  created_by?: string;
  created_at?: number;
  last_updated_at?: number;
}

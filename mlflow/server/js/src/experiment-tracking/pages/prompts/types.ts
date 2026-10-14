import type { ModelEntity, ModelVersionInfoEntity } from '../../types';

/**
 * Represents a registered prompt entry. For the time being, it uses
 * registered model entity type due to same API being reused.
 */
export type RegisteredPrompt = ModelEntity;

/**
 * Represents a registered prompt version. For the time being, it reuses model version entity
 * due to API being reused.
 */
export type RegisteredPromptVersion = ModelVersionInfoEntity;

export interface RegisteredPromptsListResponse {
  registered_models?: RegisteredPrompt[];
  next_page_token?: string;
}

export type RegisteredPromptDetailsResponse = {
  prompt?: RegisteredPrompt;
  versions: RegisteredPromptVersion[];
};

export type PromptVersionsForRunResponse = {
  model_versions?: RegisteredPromptVersion[];
};

export interface ChatPromptMessage {
  role: string;
  content: string;
}

/**
 * Represents a prompt model configuration, in the backend format (snake_case).
 */
export interface PromptModelConfig {
  provider?: string;
  model_name?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop_sequences?: string[];
  extra_params?: Record<string, any>;
}

/**
 * Represents a prompt model configuration, in the UI form format (camelCase with string inputs).
 */
export interface PromptModelConfigFormData {
  provider?: string;
  modelName?: string;
  temperature?: string;
  maxTokens?: string;
  topP?: string;
  topK?: string;
  frequencyPenalty?: string;
  presencePenalty?: string;
  stopSequences?: string;
}

// Scalar JSON types the visual schema editor can represent, on their own and as array items.
export const SCHEMA_SCALAR_TYPES = ['string', 'number', 'integer', 'boolean'] as const;

export const SCHEMA_PROPERTY_TYPES = [...SCHEMA_SCALAR_TYPES, 'enum'] as const;

export type SchemaScalarType = (typeof SCHEMA_SCALAR_TYPES)[number];
export type SchemaPropertyType = (typeof SCHEMA_PROPERTY_TYPES)[number];

interface SchemaPropertyBase {
  // Local React key, not part of the emitted schema
  id: string;
  name: string;
  required: boolean;
  // Name in the schema the editor was opened with; undefined for rows added since
  originalName?: string;
}

export interface EditableSchemaProperty extends SchemaPropertyBase {
  kind: 'editable';
  type: SchemaPropertyType;
  isArray: boolean;
  // Kept while `type` is not 'enum', so switching back restores the values
  enumValues: string[];
  // JSON type of the source enum values, so that `[1, 2]` is not re-emitted as `['1', '2']`
  enumValueType?: SchemaScalarType;
}

// A definition the flat editor cannot model (nested object, `$ref`, `oneOf`, ...), emitted verbatim
export interface ReadOnlySchemaProperty extends SchemaPropertyBase {
  kind: 'readOnly';
  label: string;
}

export type SchemaProperty = EditableSchemaProperty | ReadOnlySchemaProperty;

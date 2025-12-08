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
 * Backend format for prompt model configuration (snake_case).
 * Matches the PromptModelConfig class in mlflow/entities/model_registry/prompt_version.py
 */
export interface PromptModelConfig {
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
 * Form format for prompt model configuration (camelCase with string inputs).
 * Used for form state management with react-hook-form.
 */
export interface PromptModelConfigFormData {
  modelName?: string;
  temperature?: string; // String for form input, will be parsed to number
  maxTokens?: string; // String for form input, will be parsed to number
  topP?: string; // String for form input, will be parsed to number
  topK?: string; // String for form input, will be parsed to number
  frequencyPenalty?: string; // String for form input, will be parsed to number
  presencePenalty?: string; // String for form input, will be parsed to number
  stopSequences?: string; // Comma-separated string for form input, will be parsed to array
}

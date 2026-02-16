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

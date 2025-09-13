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

import { ModelEntity, ModelVersionInfoEntity } from '../../types';

export type RegisteredPrompt = ModelEntity;

export type RegisteredPromptVersion = ModelVersionInfoEntity;

export interface RegisteredPromptsListResponse {
  registered_models?: ModelEntity[];
  next_page_token?: string;
}

export type RegisteredPromptDetailsResponse = {
  prompt?: RegisteredPrompt;
  versions: RegisteredPromptVersion[];
};

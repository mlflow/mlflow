import type { RegisteredPrompt, RegisteredPromptVersion } from './types';

export const REGISTERED_PROMPT_CONTENT_TAG_KEY = 'mlflow.prompt.text';

export type PromptsTableMetadata = { onEditTags: (editedEntity: RegisteredPrompt) => void };

export enum PromptVersionsTableMode {
  TABLE = 'table',
  PREVIEW = 'preview',
  COMPARE = 'compare',
}

export const getPromptContentTagValue = (promptVersion: RegisteredPromptVersion) => {
  return promptVersion?.tags?.find((tag) => tag.key === REGISTERED_PROMPT_CONTENT_TAG_KEY)?.value;
};

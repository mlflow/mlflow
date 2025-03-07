import type { RegisteredPrompt, RegisteredPromptVersion } from './types';

export const REGISTERED_PROMPT_CONTENT_TAG_KEY = 'mlflow.prompt.text';
export const REGISTERED_PROMPT_COMMIT_MESSAGE_TAG_KEY = 'mlflow.prompt.commit_message';
export const REGISTERED_PROMPT_SOURCE_RUN_ID = 'mlflow.prompt.run_id';
export const IS_PROMPT_TAG_NAME = 'mlflow.prompt.is_prompt';
export const IS_PROMPT_TAG_VALUE = 'True';

export type PromptsTableMetadata = { onEditTags: (editedEntity: RegisteredPrompt) => void };
export type PromptsVersionsTableMetadata = {
  showEditAliasesModal: (versionNumber: string) => void;
  aliasesByVersion: Record<string, string[]>;
  registeredPrompt: RegisteredPrompt;
};

export enum PromptVersionsTableMode {
  TABLE = 'table',
  PREVIEW = 'preview',
  COMPARE = 'compare',
}

export const getPromptContentTagValue = (promptVersion: RegisteredPromptVersion) => {
  return promptVersion?.tags?.find((tag) => tag.key === REGISTERED_PROMPT_CONTENT_TAG_KEY)?.value;
};

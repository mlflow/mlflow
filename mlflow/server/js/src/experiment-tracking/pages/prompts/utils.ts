import type { KeyValueEntity } from '@mlflow/mlflow/src/common/types';
import type { RegisteredPrompt, RegisteredPromptVersion } from './types';

export const REGISTERED_PROMPT_CONTENT_TAG_KEY = 'mlflow.prompt.text';
// Tag key used to store the run ID associated with a single prompt version
export const REGISTERED_PROMPT_SOURCE_RUN_ID = 'mlflow.prompt.run_id';
// Tak key used to store comma-separated run IDs associated with a prompt
export const REGISTERED_PROMPT_SOURCE_RUN_IDS = 'mlflow.prompt.run_ids';
export const IS_PROMPT_TAG_NAME = 'mlflow.prompt.is_prompt';
export const IS_PROMPT_TAG_VALUE = 'true';
// Key used to store a list of prompt versions associated with a run
export const LINKED_PROMPTS_TAG_KEY = 'mlflow.linkedPrompts';

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

export const parseLinkedPromptsFromRunTags = (
  tags: Record<string, KeyValueEntity>,
): { name: string; version: string }[] => {
  const linkedPromptsTag = tags[LINKED_PROMPTS_TAG_KEY];
  if (!linkedPromptsTag || !linkedPromptsTag.value) {
    return [];
  }

  try {
    const promptVersions = JSON.parse(linkedPromptsTag.value);
    if (Array.isArray(promptVersions)) {
      return promptVersions.filter(
        (prompt: any) => prompt && typeof prompt.name === 'string' && typeof prompt.version === 'string',
      );
    }
  } catch (error) {
    console.warn('Failed to parse linked prompts from run tags:', error);
  }

  return [];
};

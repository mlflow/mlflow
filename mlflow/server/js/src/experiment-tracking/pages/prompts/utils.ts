import type { KeyValueEntity } from '@mlflow/mlflow/src/common/types';
import type { RegisteredPrompt, RegisteredPromptVersion } from './types';
import { MLFLOW_LINKED_PROMPTS_TAG } from '../../constants';

export const REGISTERED_PROMPT_CONTENT_TAG_KEY = 'mlflow.prompt.text';
// Tag key used to store the run ID associated with a single prompt version
export const REGISTERED_PROMPT_SOURCE_RUN_ID = 'mlflow.prompt.run_id';
// Tak key used to store comma-separated run IDs associated with a prompt
export const REGISTERED_PROMPT_SOURCE_RUN_IDS = 'mlflow.prompt.run_ids';
export const IS_PROMPT_TAG_NAME = 'mlflow.prompt.is_prompt';
export const IS_PROMPT_TAG_VALUE = 'true';

// Query parameter name for specifying prompt version in URLs
export const PROMPT_VERSION_QUERY_PARAM = 'promptVersion';

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
  const linkedPromptsTag = tags[MLFLOW_LINKED_PROMPTS_TAG];
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
    // do nothing, just don't display any linked prompts
  }

  return [];
};

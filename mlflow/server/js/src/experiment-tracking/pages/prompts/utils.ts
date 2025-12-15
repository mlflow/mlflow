import type { KeyValueEntity } from '@mlflow/mlflow/src/common/types';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';
import type {
  ChatPromptMessage,
  PromptModelConfig,
  PromptModelConfigFormData,
  RegisteredPrompt,
  RegisteredPromptVersion,
} from './types';
import { MLFLOW_LINKED_PROMPTS_TAG } from '../../constants';

export const REGISTERED_PROMPT_CONTENT_TAG_KEY = 'mlflow.prompt.text';
// Tag key used to store the run ID associated with a single prompt version
export const REGISTERED_PROMPT_SOURCE_RUN_ID = 'mlflow.prompt.run_id';
// Tak key used to store comma-separated run IDs associated with a prompt
export const REGISTERED_PROMPT_SOURCE_RUN_IDS = 'mlflow.prompt.run_ids';
export const IS_PROMPT_TAG_NAME = 'mlflow.prompt.is_prompt';
export const IS_PROMPT_TAG_VALUE = 'true';
export const PROMPT_TYPE_TEXT = 'text' as const;
export const PROMPT_TYPE_CHAT = 'chat' as const;
export const PROMPT_TYPE_TAG_KEY = '_mlflow_prompt_type';
// Tag key used to store comma-separated experiment IDs associated with a prompt
export const PROMPT_EXPERIMENT_IDS_TAG_KEY = '_mlflow_experiment_ids';
// Tag key used to store model config as JSON string (must match backend)
export const PROMPT_MODEL_CONFIG_TAG_KEY = '_mlflow_prompt_model_config';

export const MODEL_CONFIG_FIELD_LABELS: Record<Exclude<keyof PromptModelConfig, 'extra_params'>, string> = {
  provider: 'Provider',
  model_name: 'Model',
  temperature: 'Temperature',
  max_tokens: 'Max Tokens',
  top_p: 'Top P',
  top_k: 'Top K',
  frequency_penalty: 'Frequency Penalty',
  presence_penalty: 'Presence Penalty',
  stop_sequences: 'Stop Sequences',
} as const;

// Query parameter name for specifying prompt version in URLs
export const PROMPT_VERSION_QUERY_PARAM = 'promptVersion';

export type PromptsTableMetadata = {
  onEditTags: (editedEntity: RegisteredPrompt) => void;
  experimentId?: string;
};
export type PromptsVersionsTableMetadata = {
  showEditAliasesModal: (versionNumber: string) => void;
  aliasesByVersion: Record<string, string[]>;
  registeredPrompt: RegisteredPrompt;
};

export enum PromptVersionsTableMode {
  PREVIEW = 'preview',
  COMPARE = 'compare',
}

export const getPromptContentTagValue = (promptVersion: RegisteredPromptVersion) => {
  return promptVersion?.tags?.find((tag) => tag.key === REGISTERED_PROMPT_CONTENT_TAG_KEY)?.value;
};

export const isChatPrompt = (promptVersion?: RegisteredPromptVersion): boolean => {
  const tagValue = promptVersion?.tags?.find((tag) => tag.key === PROMPT_TYPE_TAG_KEY)?.value;
  return tagValue === PROMPT_TYPE_CHAT;
};

const isPromptChatMessage = (value: any): value is ChatPromptMessage => {
  return value && typeof value === 'object' && typeof value.role === 'string' && typeof value.content === 'string';
};

const isPromptChatMessages = (value: unknown): value is ChatPromptMessage[] => {
  return Array.isArray(value) && value.every((item) => isPromptChatMessage(item));
};

export const getChatPromptMessagesFromValue = (value?: string) => {
  if (!value) {
    return undefined;
  }

  const parsedValue = parseJSONSafe(value);
  if (isPromptChatMessages(parsedValue)) {
    return parsedValue;
  }

  return undefined;
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

/**
 * Builds a filter clause from a search string.
 * If the search string contains SQL-like keywords (ILIKE, LIKE, =, !=),
 * it's treated as a raw filter query. Otherwise, it's treated as a simple
 * name search and wrapped with ILIKE pattern matching.
 *
 * @param searchFilter - The search string to process
 * @returns The filter clause, or an empty string if searchFilter is empty
 */
export const buildSearchFilterClause = (searchFilter?: string): string => {
  if (!searchFilter) {
    return '';
  }

  // Check if the search filter looks like a SQL-like query
  // If so, treat it as a raw filter query; otherwise, treat it as a simple name search
  const sqlKeywordPattern = /(\s+(ILIKE|LIKE)\s+)|=|!=/i;

  if (sqlKeywordPattern.test(searchFilter)) {
    // User provided a SQL-like filter, use it directly
    return searchFilter;
  } else {
    // Simple name search
    return `name ILIKE '%${searchFilter}%'`;
  }
};

/**
 * Parse model config from tag value (JSON string).
 * Returns undefined if tag doesn't exist or JSON parsing fails.
 */
export const getModelConfigFromTags = (tags?: KeyValueEntity[]): PromptModelConfig | undefined => {
  const configTag = tags?.find((tag) => tag.key === PROMPT_MODEL_CONFIG_TAG_KEY);
  if (!configTag?.value) {
    return undefined;
  }

  try {
    return JSON.parse(configTag.value) as PromptModelConfig;
  } catch (error) {
    console.error('Failed to parse model config:', error);
    return undefined;
  }
};

/**
 * Convert form data to backend model config format.
 * Returns undefined if no fields have values.
 */
export const formDataToModelConfig = (formData: PromptModelConfigFormData): PromptModelConfig | undefined => {
  const hasAnyValue = Object.values(formData).some((v) => !!v);
  if (!hasAnyValue) {
    return undefined;
  }

  const config: PromptModelConfig = {};

  const provider = formData.provider?.trim();
  if (provider) {
    config.provider = provider;
  }
  const modelName = formData.modelName?.trim();
  if (modelName) {
    config.model_name = modelName;
  }
  if (formData.temperature?.trim()) {
    const temp = parseFloat(formData.temperature);
    if (!isNaN(temp)) config.temperature = temp;
  }
  if (formData.maxTokens?.trim()) {
    const tokens = parseInt(formData.maxTokens, 10);
    if (!isNaN(tokens)) config.max_tokens = tokens;
  }
  if (formData.topP?.trim()) {
    const topP = parseFloat(formData.topP);
    if (!isNaN(topP)) config.top_p = topP;
  }
  if (formData.topK?.trim()) {
    const topK = parseInt(formData.topK, 10);
    if (!isNaN(topK)) config.top_k = topK;
  }
  if (formData.frequencyPenalty?.trim()) {
    const freqPenalty = parseFloat(formData.frequencyPenalty);
    if (!isNaN(freqPenalty)) config.frequency_penalty = freqPenalty;
  }
  if (formData.presencePenalty?.trim()) {
    const presPenalty = parseFloat(formData.presencePenalty);
    if (!isNaN(presPenalty)) config.presence_penalty = presPenalty;
  }
  if (formData.stopSequences?.trim()) {
    // Split by comma and trim each value
    const sequences = formData.stopSequences
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    if (sequences.length > 0) config.stop_sequences = sequences;
  }

  return Object.keys(config).length > 0 ? config : undefined;
};

/**
 * Convert backend model config to form data format.
 * Returns empty object if config is undefined.
 */
export const modelConfigToFormData = (config?: PromptModelConfig): PromptModelConfigFormData => {
  if (!config) {
    return {};
  }

  return {
    provider: config.provider ?? '',
    modelName: config.model_name ?? '',
    temperature: config.temperature !== undefined ? String(config.temperature) : '',
    maxTokens: config.max_tokens !== undefined ? String(config.max_tokens) : '',
    topP: config.top_p !== undefined ? String(config.top_p) : '',
    topK: config.top_k !== undefined ? String(config.top_k) : '',
    frequencyPenalty: config.frequency_penalty !== undefined ? String(config.frequency_penalty) : '',
    presencePenalty: config.presence_penalty !== undefined ? String(config.presence_penalty) : '',
    stopSequences: config.stop_sequences ? config.stop_sequences.join(', ') : '',
  };
};

/**
 * Validate model config form values.
 * Returns an object with field names as keys and error messages as values.
 */
export const validateModelConfig = (formData: PromptModelConfigFormData): Record<string, string> => {
  const errors: Record<string, string> = {};

  if (formData['temperature']?.trim()) {
    const temp = parseFloat(formData['temperature']);
    if (isNaN(temp) || temp < 0) {
      errors['temperature'] = 'Temperature must be a number >= 0';
    }
  }

  if (formData['maxTokens']?.trim()) {
    const tokens = parseInt(formData['maxTokens'], 10);
    if (isNaN(tokens) || tokens <= 0) {
      errors['maxTokens'] = 'Max tokens must be an integer > 0';
    }
  }

  if (formData['topP']?.trim()) {
    const topP = parseFloat(formData['topP']);
    if (isNaN(topP) || topP < 0 || topP > 1) {
      errors['topP'] = 'Top P must be a number between 0 and 1';
    }
  }

  if (formData['topK']?.trim()) {
    const topK = parseInt(formData['topK'], 10);
    if (isNaN(topK) || topK <= 0) {
      errors['topK'] = 'Top K must be an integer > 0';
    }
  }

  if (formData['frequencyPenalty']?.trim()) {
    const freqPenalty = parseFloat(formData['frequencyPenalty']);
    if (isNaN(freqPenalty) || freqPenalty < -2 || freqPenalty > 2) {
      errors['frequencyPenalty'] = 'Frequency penalty must be a number between -2 and 2';
    }
  }

  if (formData['presencePenalty']?.trim()) {
    const presPenalty = parseFloat(formData['presencePenalty']);
    if (isNaN(presPenalty) || presPenalty < -2 || presPenalty > 2) {
      errors['presencePenalty'] = 'Presence penalty must be a number between -2 and 2';
    }
  }

  return errors;
};

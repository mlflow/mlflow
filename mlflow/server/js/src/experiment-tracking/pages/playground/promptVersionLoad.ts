import { getChatPromptMessagesFromValue, getPromptContentTagValue, isChatPrompt } from '../prompts/utils';
import { getModelConfigFromTags, getResponseFormatFromTags } from '../prompts/utils';
import type { RegisteredPromptVersion } from '../prompts/types';
import type { ChatMessage, ChatRole, PlaygroundParams, ResponseFormatType } from './types';

const KNOWN_ROLES: ChatRole[] = ['system', 'user', 'assistant'];

const isKnownRole = (role: string): role is ChatRole => (KNOWN_ROLES as string[]).includes(role);

export const buildMessagesFromVersion = (version: RegisteredPromptVersion): ChatMessage[] => {
  const tagValue = getPromptContentTagValue(version) ?? '';

  if (isChatPrompt(version)) {
    const parsed = getChatPromptMessagesFromValue(tagValue);
    if (!parsed) {
      return [];
    }
    return parsed.map((message) => ({
      role: isKnownRole(message.role) ? message.role : 'user',
      content: message.content,
    }));
  }

  return [{ role: 'user', content: tagValue }];
};

export interface PromptLoadSettings {
  params: PlaygroundParams;
  responseFormatType: ResponseFormatType;
  responseFormatSchemaText: string;
}

export interface PromptLoadPayload {
  messages: ChatMessage[];
  settings: PromptLoadSettings | null;
  promptName: string;
  versionLabel: string;
}

export const buildLoadPayloadFromVersion = (version: RegisteredPromptVersion): PromptLoadPayload => {
  const messages = buildMessagesFromVersion(version);
  const modelConfig = getModelConfigFromTags(version.tags);
  const responseFormatRaw = getResponseFormatFromTags(version.tags);
  const hasResponseFormat = typeof responseFormatRaw === 'string' && responseFormatRaw.trim().length > 0;

  if (!modelConfig && !hasResponseFormat) {
    return {
      messages,
      settings: null,
      promptName: version.name,
      versionLabel: version.version,
    };
  }

  const params: PlaygroundParams = {};
  if (modelConfig) {
    if (modelConfig.temperature !== undefined) params.temperature = modelConfig.temperature;
    if (modelConfig.max_tokens !== undefined) params.max_tokens = modelConfig.max_tokens;
    if (modelConfig.top_p !== undefined) params.top_p = modelConfig.top_p;
    if (modelConfig.top_k !== undefined) params.top_k = modelConfig.top_k;
    if (modelConfig.frequency_penalty !== undefined) params.frequency_penalty = modelConfig.frequency_penalty;
    if (modelConfig.presence_penalty !== undefined) params.presence_penalty = modelConfig.presence_penalty;
    if (Array.isArray(modelConfig.stop_sequences) && modelConfig.stop_sequences.length > 0) {
      params.stop = modelConfig.stop_sequences;
    }
  }

  return {
    messages,
    settings: {
      params,
      responseFormatType: hasResponseFormat ? 'json_schema' : 'text',
      responseFormatSchemaText: hasResponseFormat ? (responseFormatRaw as string) : '',
    },
    promptName: version.name,
    versionLabel: version.version,
  };
};

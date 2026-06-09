import type { PromptModelConfig } from '../prompts/types';
import type { ChatMessage, PlaygroundParams, ResponseFormatType } from './types';

/**
 * Convert the playground's request params into the registry's model-config
 * shape (snake_case, matching `PromptModelConfig`). The playground exposes
 * `stop`, which the registry stores as `stop_sequences`. Provider/model name
 * are not tracked here because the endpoint is selected separately and is not
 * part of a saved prompt template. Returns undefined when nothing is set.
 */
export const paramsToModelConfig = (params: PlaygroundParams): PromptModelConfig | undefined => {
  const config: PromptModelConfig = {};
  if (params.temperature !== undefined) config.temperature = params.temperature;
  if (params.max_tokens !== undefined) config.max_tokens = params.max_tokens;
  if (params.top_p !== undefined) config.top_p = params.top_p;
  if (params.top_k !== undefined) config.top_k = params.top_k;
  if (params.frequency_penalty !== undefined) config.frequency_penalty = params.frequency_penalty;
  if (params.presence_penalty !== undefined) config.presence_penalty = params.presence_penalty;
  if (params.stop && params.stop.length > 0) config.stop_sequences = params.stop;
  return Object.keys(config).length > 0 ? config : undefined;
};

/**
 * The messages that will actually be written to the new version: content is
 * trimmed and empty messages are dropped. The playground always keeps a
 * trailing empty user turn, so this prevents that placeholder (and any blank
 * turns) from being persisted as part of the template.
 */
export const getSaveableMessages = (messages: ChatMessage[]): ChatMessage[] =>
  messages
    .map((message) => ({ role: message.role, content: message.content.trim() }))
    .filter((message) => message.content.length > 0);

/**
 * True when the playground state carries model settings worth persisting
 * alongside the prompt (model config and/or a JSON-schema response format).
 */
export const hasSaveableSettings = (
  params: PlaygroundParams,
  responseFormatType: ResponseFormatType,
  responseFormatSchemaText: string,
): boolean =>
  paramsToModelConfig(params) !== undefined ||
  (responseFormatType === 'json_schema' && responseFormatSchemaText.trim().length > 0);

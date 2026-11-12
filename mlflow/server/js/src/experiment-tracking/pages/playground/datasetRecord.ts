import { findLastIndex, isEqual } from 'lodash';
import { MLFLOW_LINKED_PROMPTS_TAG } from '../../constants';
import type { ChatMessage, ConversationMessage } from './types';
import { buildRequestMessages, substituteVariables } from './utils';

const hasContentAfterIndex = (messages: ConversationMessage[], index: number): boolean =>
  messages.slice(index + 1).some((message) => (message.content ?? '').trim().length > 0);

/**
 * Index of the reply that answers the current prompt, or -1 when there is none. Only a
 * model reply counts, and only while nothing but the empty next-turn composer follows it.
 * An assistant turn the user wrote or loaded from the registry (a prefill or a few-shot
 * example) is part of the prompt, and once the user types a new turn the reply becomes
 * conversation context.
 */
const findReplyIndex = (messages: ConversationMessage[]): number => {
  const lastAssistantIndex = findLastIndex(messages, (message) => message.role === 'assistant');
  if (lastAssistantIndex < 0 || !messages[lastAssistantIndex].isModelReply) {
    return -1;
  }
  return hasContentAfterIndex(messages, lastAssistantIndex) ? -1 : lastAssistantIndex;
};

/** The turns that make up the record's inputs, before variables are resolved. */
export const getInputTurns = (messages: ConversationMessage[]): ConversationMessage[] => {
  const replyIndex = findReplyIndex(messages);
  return replyIndex >= 0 ? messages.slice(0, replyIndex) : messages;
};

/**
 * The record's input messages, with `{{ variables }}` resolved and empty turns dropped.
 * Each turn keeps `role`, `content` and any `tool_calls`, so an assistant turn that only
 * called a tool stays in the conversation. Display-only fields such as `usage` are dropped.
 */
export const getDatasetInputMessages = (
  messages: ConversationMessage[],
  variables: Record<string, string>,
): ChatMessage[] =>
  substituteVariables(getInputTurns(messages), variables)
    .filter((message) => (message.content ?? '').trim().length > 0 || Boolean(message.tool_calls?.length))
    .map(({ role, content, tool_calls }) => ({ role, content, ...(tool_calls?.length ? { tool_calls } : {}) }));

export interface LatestReply {
  /** Text to pre-fill as the expected response, or `''` when there is nothing to pre-fill. */
  content: string;
  /** True when the prompt changed after the reply was generated, so the reply is withheld. */
  isOutdated: boolean;
}

/**
 * The generated reply to pre-fill as the expected response. It is withheld when the inputs
 * no longer match the request that produced it (for example, a variable was changed after
 * the run), so a record never pairs new inputs with the answer to old ones.
 */
export const getLatestReply = (messages: ConversationMessage[], variables: Record<string, string>): LatestReply => {
  const replyIndex = findReplyIndex(messages);
  const content = replyIndex >= 0 ? (messages[replyIndex].content ?? '') : '';
  if (!content.trim()) {
    return { content: '', isOutdated: false };
  }
  const currentRequest = buildRequestMessages(messages.slice(0, replyIndex), variables);
  if (!isEqual(currentRequest, messages[replyIndex].requestMessages)) {
    return { content: '', isOutdated: true };
  }
  return { content, isOutdated: false };
};

/** A record in the shape the `ajax-api/3.0/mlflow/datasets/{id}/records` upsert accepts. */
export interface PlaygroundDatasetRecord {
  inputs: { messages: ChatMessage[] };
  expectations?: { expected_response: string };
  source: { source_type: 'HUMAN'; source_data: Record<string, never> };
  tags?: Record<string, string>;
}

/**
 * Builds the evaluation-dataset record from playground state. Inputs are stored under
 * the single-turn `{ messages: [...] }` schema the datasets UI and built-in judges
 * expect. A non-empty expected response is stored under `expectations.expected_response`,
 * the key `Correctness` and other reference-based judges read; without one the
 * expectations field is omitted, so the record stays valid for reference-free scorers.
 * The record is marked as human-authored, and is linked to the registry prompt version
 * loaded in the playground, if any, through the same `mlflow.linkedPrompts` tag traces use.
 */
export const buildPlaygroundDatasetRecord = ({
  inputMessages,
  expectedResponse,
  linkedPrompt,
}: {
  inputMessages: ChatMessage[];
  expectedResponse: string;
  linkedPrompt?: { name: string; version: string } | null;
}): PlaygroundDatasetRecord => {
  const trimmed = expectedResponse.trim();
  return {
    inputs: { messages: inputMessages },
    ...(trimmed ? { expectations: { expected_response: trimmed } } : {}),
    source: { source_type: 'HUMAN', source_data: {} },
    ...(linkedPrompt
      ? {
          tags: {
            [MLFLOW_LINKED_PROMPTS_TAG]: JSON.stringify([{ name: linkedPrompt.name, version: linkedPrompt.version }]),
          },
        }
      : {}),
  };
};

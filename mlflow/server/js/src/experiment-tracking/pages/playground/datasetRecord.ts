import type { DatasetRecord } from '../experiment-evaluation-datasets-v2/hooks/useDatasetsQueries';
import type { ChatMessage, ConversationMessage } from './types';
import { substituteVariables } from './utils';

const findLastAssistantIndex = (messages: ConversationMessage[]): number => {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'assistant') {
      return i;
    }
  }
  return -1;
};

const hasContentAfterIndex = (messages: ConversationMessage[], index: number): boolean =>
  messages.slice(index + 1).some((message) => (message.content ?? '').trim().length > 0);

/**
 * The latest assistant reply is split off as the record's expected response only while
 * it is still the end of the conversation (nothing but the empty next-turn composer
 * after it). Once the user types a new turn, the reply stops being "the answer to the
 * current prompt": it becomes conversation context inside the inputs instead, and no
 * expected response is pre-filled.
 */
const isLatestAssistantTheExpectation = (messages: ConversationMessage[], lastAssistantIndex: number): boolean =>
  lastAssistantIndex >= 0 && !hasContentAfterIndex(messages, lastAssistantIndex);

/**
 * Content of the assistant reply that answers the current prompt, or `''` when there is
 * none — either the prompt has not been run yet, or the user has already typed a new
 * turn after the last reply (so that reply no longer matches the prompt being saved).
 * Used to pre-fill the editable "expected response" field.
 */
export const getLatestAssistantContent = (messages: ConversationMessage[]): string => {
  const lastAssistantIndex = findLastAssistantIndex(messages);
  if (!isLatestAssistantTheExpectation(messages, lastAssistantIndex)) {
    return '';
  }
  return messages[lastAssistantIndex].content ?? '';
};

/**
 * The turns that make up the model input for a dataset record, with `{{ variables }}`
 * resolved to their values. When the conversation ends at the latest assistant reply,
 * that reply is excluded — it is captured separately as the expected response. When the
 * user has typed a new turn after the reply, the whole conversation (reply included, as
 * multi-turn context) is the input. Empty-content turns are filtered out, and each turn
 * is reduced to `{ role, content }` (display-only fields such as `usage` are stripped).
 */
export const getDatasetInputMessages = (
  messages: ConversationMessage[],
  variables: Record<string, string>,
): ChatMessage[] => {
  const lastAssistantIndex = findLastAssistantIndex(messages);
  const inputTurns = isLatestAssistantTheExpectation(messages, lastAssistantIndex)
    ? messages.slice(0, lastAssistantIndex)
    : messages;
  return substituteVariables(inputTurns, variables)
    .filter((message) => (message.content ?? '').trim().length > 0)
    .map(({ role, content }) => ({ role, content }));
};

/**
 * Builds the evaluation-dataset record from playground state. Inputs are stored under
 * the single-turn `{ messages: [...] }` schema the datasets UI and built-in judges
 * expect; a non-empty expected response is stored under `expectations.expected_response`
 * — the key `Correctness` and other reference-based judges read. The expectations field
 * is omitted entirely when no reference answer is provided, so the record stays valid for
 * reference-free scorers.
 */
export const buildPlaygroundDatasetRecord = ({
  inputMessages,
  expectedResponse,
}: {
  inputMessages: ChatMessage[];
  expectedResponse: string;
}): Partial<DatasetRecord> => {
  const trimmed = expectedResponse.trim();
  return {
    inputs: { messages: inputMessages },
    ...(trimmed ? { expectations: { expected_response: trimmed } } : {}),
  };
};

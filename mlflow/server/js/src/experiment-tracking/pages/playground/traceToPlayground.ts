import { parseModelTraceToTreeWithMultipleRoots } from '@databricks/web-shared/model-trace-explorer';
import type {
  ModelTrace,
  ModelTraceChatMessage,
  ModelTraceSpanNode,
} from '@databricks/web-shared/model-trace-explorer';

import type {
  ChatRole,
  ConversationMessage,
  PlaygroundParams,
  PlaygroundTool,
  ResponseFormatType,
  ToolChoice,
} from './types';
import { BLANK_JSON_SCHEMA } from './utils';

// A playground tool without its client-side id; the id is assigned by the page when applying the
// prefill (see PlaygroundPage), consistent with how manually-added tools get their ids.
type PrefillTool = Omit<PlaygroundTool, 'id'>;

// A tool execution captured in the trace: the call (name + arguments) and the result it returned.
// Used to answer the model's tool calls automatically on re-run instead of asking the user to
// retype outputs the trace already recorded.
export interface PrefillToolResult {
  name: string;
  args?: string;
  result: string;
}

export interface TracePlaygroundPrefill {
  // The captured input (system prompt + user turns, with prior assistant turns as multi-turn
  // context), ready to edit and re-run. The model's output rounds are intentionally excluded.
  messages: ConversationMessage[];
  // The model / gateway endpoint recorded on the span, if any. Best-effort — the captured model
  // may not map to a configured gateway endpoint, in which case the user picks one.
  endpointName: string;
  params: PlaygroundParams;
  // The tools available in the trace, so the playground restores them for re-running.
  tools: PrefillTool[];
  // Tool executions captured in the trace, used to auto-answer the model's tool calls on re-run.
  toolResults: PrefillToolResult[];
  // Tool-choice mode captured on the request; the playground supports 'auto' | 'required'.
  toolChoice: ToolChoice;
  // The response format the request used, restored so the playground re-runs it identically.
  responseFormatType: ResponseFormatType;
  responseFormatSchemaText: string;
  // The span the prefill was derived from, for the "loaded from trace" banner.
  spanName?: string;
}

const NUMERIC_PARAM_KEYS = [
  'temperature',
  'max_tokens',
  'top_p',
  'top_k',
  'presence_penalty',
  'frequency_penalty',
] as const;

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

// Roles the playground can edit. Trace messages with other roles (tool/function results) are
// dropped from the prefill since the playground models a system/user/assistant conversation.
const mapRole = (role: ModelTraceChatMessage['role']): ChatRole | null => {
  switch (role) {
    case 'system':
    case 'developer':
      return 'system';
    case 'user':
      return 'user';
    case 'assistant':
      return 'assistant';
    default:
      return null;
  }
};

const coerceContent = (content: ModelTraceChatMessage['content']): string => {
  if (typeof content === 'string') {
    return content;
  }
  if (content === null || content === undefined) {
    return '';
  }
  // Multimodal / structured content — surface it as JSON so nothing is silently lost.
  return JSON.stringify(content, null, 2);
};

const flattenNodes = (nodes: ModelTraceSpanNode[]): ModelTraceSpanNode[] => {
  const result: ModelTraceSpanNode[] = [];
  const visit = (node: ModelTraceSpanNode) => {
    result.push(node);
    node.children?.forEach(visit);
  };
  nodes.forEach(visit);
  return result;
};

// Picks the span to prefill from. For an explicit span id (span-level "Open in Playground") that
// span is used. Otherwise (trace-level) the span with the most complete conversation is used:
// agent frameworks spread a single run across many spans, and the fullest one carries the final
// response. Returns null when no span carries chat messages we can turn into a prompt.
const pickSpanNode = (flat: ModelTraceSpanNode[], spanId?: string): ModelTraceSpanNode | null => {
  if (spanId) {
    return flat.find((node) => String(node.key) === spanId) ?? null;
  }
  const withMessages = flat.filter((node) => (node.chatMessages?.length ?? 0) > 0);
  if (withMessages.length === 0) {
    return null;
  }
  return withMessages.reduce((best, node) =>
    (node.chatMessages?.length ?? 0) > (best.chatMessages?.length ?? 0) ? node : best,
  );
};

const extractParams = (inputs: unknown): PlaygroundParams => {
  const record = asRecord(inputs);
  if (!record) {
    return {};
  }
  const params: PlaygroundParams = {};
  for (const key of NUMERIC_PARAM_KEYS) {
    const value = record[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      params[key] = value;
    }
  }
  const stop = record['stop'];
  if (typeof stop === 'string') {
    params.stop = [stop];
  } else if (Array.isArray(stop) && stop.every((entry) => typeof entry === 'string')) {
    params.stop = stop as string[];
  }
  return params;
};

// Aggregates the tool definitions available anywhere in the trace, de-duplicated by tool name.
// Tools are read from the shared parser's normalized `chatTools`, which the model-trace-explorer
// extracts per span from the `mlflow.chat.tools` attribute or — shape-based, framework-agnostic —
// from the span's request inputs. Aggregating across all spans matters for agent runs, which
// record tools on their individual model-request spans rather than the root span.
const extractTools = (flat: ModelTraceSpanNode[]): PrefillTool[] => {
  const byName = new Map<string, PrefillTool>();
  for (const node of flat) {
    for (const tool of node.chatTools ?? []) {
      const name = tool.function?.name;
      if (!name || byName.has(name)) {
        continue;
      }
      byName.set(name, {
        name,
        description: tool.function.description ?? '',
        params: tool.function.parameters ? JSON.stringify(tool.function.parameters, null, 2) : BLANK_JSON_SCHEMA,
      });
    }
  }
  return [...byName.values()];
};

// Pairs each captured assistant tool call with the tool result that answered it, walking the
// normalized conversation of every span. Pairing uses tool_call_id when both sides carry it and
// falls back to call order otherwise (not every framework normalizer preserves ids). Duplicate
// executions (same call, same result — common because each agent round re-records the history)
// are de-duplicated.
const extractToolResults = (flat: ModelTraceSpanNode[]): PrefillToolResult[] => {
  const seen = new Set<string>();
  const results: PrefillToolResult[] = [];
  for (const node of flat) {
    const pendingCalls: { id?: string; name: string; args?: string }[] = [];
    for (const message of node.chatMessages ?? []) {
      if (message.role === 'assistant' && Array.isArray(message.tool_calls)) {
        for (const toolCall of message.tool_calls) {
          if (toolCall.function?.name) {
            pendingCalls.push({ id: toolCall.id, name: toolCall.function.name, args: toolCall.function.arguments });
          }
        }
      } else if (message.role === 'tool') {
        let call: { id?: string; name: string; args?: string } | undefined;
        if (message.tool_call_id) {
          // An id that matches no pending call is an orphan — skip it rather than mispairing it
          // with an unrelated call positionally.
          const index = pendingCalls.findIndex((pending) => pending.id === message.tool_call_id);
          call = index >= 0 ? pendingCalls.splice(index, 1)[0] : undefined;
        } else {
          call = pendingCalls.shift();
        }
        const result = coerceContent(message.content);
        if (!call || !result) {
          continue;
        }
        const key = `${call.name}|${call.args ?? ''}|${result}`;
        if (!seen.has(key)) {
          seen.add(key);
          results.push({ name: call.name, ...(call.args !== undefined ? { args: call.args } : {}), result });
        }
      }
    }
  }
  return results;
};

const extractToolChoice = (inputs: unknown): ToolChoice =>
  asRecord(inputs)?.['tool_choice'] === 'required' ? 'required' : 'auto';

const extractResponseFormat = (inputs: unknown): { type: ResponseFormatType; schemaText: string } => {
  const format = asRecord(asRecord(inputs)?.['response_format']);
  if (format?.['type'] === 'json_object') {
    return { type: 'json_object', schemaText: '' };
  }
  if (format?.['type'] === 'json_schema') {
    const schema = asRecord(format['json_schema'])?.['schema'];
    return { type: 'json_schema', schemaText: schema ? JSON.stringify(schema, null, 2) : '' };
  }
  return { type: 'text', schemaText: '' };
};

// Maps the normalized chat messages into the playground prefill: the captured *input* only.
// Everything after the last user turn — assistant tool-call rounds, tool results, and the final
// reply — is the model's output being debugged; the user re-runs the input (with the restored
// tools/params) to regenerate it. Assistant turns *before* the last user turn are genuine
// multi-turn context and are kept as plain text; tool-result turns and empty assistant turns are
// execution noise and dropped.
const toConversationMessages = (chatMessages: ModelTraceChatMessage[]): ConversationMessage[] => {
  let lastUserIndex = -1;
  for (let i = chatMessages.length - 1; i >= 0; i -= 1) {
    if (chatMessages[i].role === 'user') {
      lastUserIndex = i;
      break;
    }
  }
  const inputMessages = lastUserIndex >= 0 ? chatMessages.slice(0, lastUserIndex + 1) : chatMessages;

  const mapped: ConversationMessage[] = [];
  for (const message of inputMessages) {
    const role = mapRole(message.role);
    if (role === null || role === 'tool') {
      continue;
    }
    const content = coerceContent(message.content);
    if (role === 'assistant' && !content) {
      continue;
    }
    mapped.push({ role, content });
  }
  return mapped;
};

/**
 * Builds the playground prefill (messages, model, params, tools, tool choice, response format)
 * from a fetched trace, working across framework integrations: messages are read from the shared
 * parser's normalized `chatMessages` (which handles OpenAI, PydanticAI, LangChain, Anthropic, …),
 * and tools are collected from every span so agent runs — which record tools on their individual
 * model-request spans — are covered too.
 *
 * @param modelTrace The full trace (info + spans) fetched by id.
 * @param spanId Optional span id to prefill from (span-level "Open in Playground"). When omitted,
 *   the span with the most complete conversation is used (trace-level "Open in Playground").
 * @returns The prefill, or null when the trace has no chat/LLM span to derive a prompt from.
 */
export const buildPlaygroundPrefillFromTrace = (
  modelTrace: ModelTrace,
  spanId?: string,
): TracePlaygroundPrefill | null => {
  let nodes: ModelTraceSpanNode[];
  try {
    nodes = parseModelTraceToTreeWithMultipleRoots(modelTrace);
  } catch {
    return null;
  }

  const flat = flattenNodes(nodes);
  const node = pickSpanNode(flat, spanId);
  if (!node || !(node.chatMessages?.length ?? 0)) {
    return null;
  }

  const messages = toConversationMessages(node.chatMessages ?? []);
  const responseFormat = extractResponseFormat(node.inputs);
  return {
    messages: messages.length > 0 ? messages : [{ role: 'user', content: '' }],
    endpointName: typeof node.modelName === 'string' ? node.modelName : '',
    params: extractParams(node.inputs),
    tools: extractTools(flat),
    toolResults: extractToolResults(flat),
    toolChoice: extractToolChoice(node.inputs),
    responseFormatType: responseFormat.type,
    responseFormatSchemaText: responseFormat.schemaText,
    spanName: typeof node.title === 'string' ? node.title : undefined,
  };
};

import { compact, has, isArray, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceSpanNode, ModelTraceToolCall } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

/**
 * VoltAgent Message Format:
 *
 * Messages are in the `agent.messages` or `llm.messages` attribute as a JSON array.
 * Each message has:
 * - role: "system" | "user" | "assistant" | "tool"
 * - content: string | ContentPart[]
 *
 * ContentPart can be:
 * - { type: "text", text: string }
 * - { type: "tool-call", toolCallId: string, toolName: string, input: object }
 * - { type: "tool-result", toolCallId: string, toolName: string, output: object }
 */

type VoltAgentTextContent = {
  type: 'text';
  text: string;
};

type VoltAgentToolCallContent = {
  type: 'tool-call';
  toolCallId: string;
  toolName: string;
  input: any;
  providerExecuted?: boolean;
};

type VoltAgentToolResultContent = {
  type: 'tool-result';
  toolCallId: string;
  toolName: string;
  output: any;
};

type VoltAgentContentPart = VoltAgentTextContent | VoltAgentToolCallContent | VoltAgentToolResultContent;

type VoltAgentMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | VoltAgentContentPart[];
};

const isVoltAgentTextContent = (obj: unknown): obj is VoltAgentTextContent => {
  if (!isObject(obj)) return false;
  const typedObj = obj as any;
  return typedObj.type === 'text' && isString(typedObj.text);
};

const isVoltAgentToolCallContent = (obj: unknown): obj is VoltAgentToolCallContent => {
  if (!isObject(obj)) return false;
  const typedObj = obj as any;
  return typedObj.type === 'tool-call' && isString(typedObj.toolCallId) && isString(typedObj.toolName);
};

const isVoltAgentToolResultContent = (obj: unknown): obj is VoltAgentToolResultContent => {
  if (!isObject(obj)) return false;
  const typedObj = obj as any;
  return typedObj.type === 'tool-result' && isString(typedObj.toolCallId) && isString(typedObj.toolName);
};

const isVoltAgentContentPart = (obj: unknown): obj is VoltAgentContentPart => {
  return isVoltAgentTextContent(obj) || isVoltAgentToolCallContent(obj) || isVoltAgentToolResultContent(obj);
};

const isVoltAgentMessage = (obj: unknown): obj is VoltAgentMessage => {
  if (!isObject(obj)) return false;
  const typedObj = obj as any;

  const hasRole =
    has(obj, 'role') && isString(typedObj.role) && ['system', 'user', 'assistant', 'tool'].includes(typedObj.role);

  if (!hasRole) return false;

  // Content can be a string or an array of content parts
  if (isString(typedObj.content)) return true;
  if (isArray(typedObj.content)) {
    return typedObj.content.every(isVoltAgentContentPart);
  }

  return false;
};

const extractToolCalls = (content: VoltAgentContentPart[]): ModelTraceToolCall[] => {
  const toolCalls: ModelTraceToolCall[] = [];

  for (const part of content) {
    if (isVoltAgentToolCallContent(part)) {
      toolCalls.push({
        id: part.toolCallId,
        function: {
          name: part.toolName,
          arguments: JSON.stringify(part.input, null, 2),
        },
      });
    }
  }

  return toolCalls;
};

const extractTextContent = (content: VoltAgentContentPart[]): string => {
  const textParts: string[] = [];

  for (const part of content) {
    if (isVoltAgentTextContent(part)) {
      textParts.push(part.text);
    }
  }

  return textParts.join('\n\n');
};

const extractToolResultContent = (content: VoltAgentContentPart[]): string | undefined => {
  for (const part of content) {
    if (isVoltAgentToolResultContent(part)) {
      const output = part.output;
      // Handle nested output structure
      if (output?.type === 'json' && output?.value) {
        return JSON.stringify(output.value, null, 2);
      }
      return JSON.stringify(output, null, 2);
    }
  }
  return undefined;
};

const extractToolCallId = (content: VoltAgentContentPart[]): string | undefined => {
  for (const part of content) {
    if (isVoltAgentToolResultContent(part)) {
      return part.toolCallId;
    }
  }
  return undefined;
};

const processVoltAgentMessage = (message: VoltAgentMessage): ModelTraceChatMessage | null => {
  if (isString(message.content)) {
    return prettyPrintChatMessage({
      role: message.role,
      content: message.content,
    });
  }

  const contentParts = message.content;

  if (message.role === 'tool') {
    const toolResultContent = extractToolResultContent(contentParts);
    const toolCallId = extractToolCallId(contentParts);

    return prettyPrintChatMessage({
      role: 'tool',
      content: toolResultContent,
      ...(toolCallId && { tool_call_id: toolCallId }),
    });
  }

  if (message.role === 'assistant') {
    const textContent = extractTextContent(contentParts);
    const toolCalls = extractToolCalls(contentParts);

    return prettyPrintChatMessage({
      role: 'assistant',
      content: textContent || undefined,
      ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
    });
  }

  const textContent = extractTextContent(contentParts);

  return prettyPrintChatMessage({
    role: message.role,
    content: textContent,
  });
};

export const normalizeVoltAgentChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  let messages: unknown[] | null = null;

  if (isString(obj)) {
    try {
      const parsed = JSON.parse(obj);
      if (isArray(parsed)) {
        messages = parsed;
      }
    } catch {
      return null;
    }
  } else if (isArray(obj)) {
    messages = obj;
  }

  if (!messages || messages.length === 0) {
    return null;
  }

  if (!messages.every(isVoltAgentMessage)) {
    return null;
  }

  return compact((messages as VoltAgentMessage[]).map(processVoltAgentMessage));
};

export const normalizeVoltAgentChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (isString(obj) && obj.trim().length > 0) {
    const message = prettyPrintChatMessage({
      role: 'assistant',
      content: obj,
    });
    return message ? [message] : null;
  }

  return null;
};

const getToolCallIdFromSpan = (child: ModelTraceSpanNode): string => {
  if (Array.isArray(child.attributes)) {
    const attribute = child.attributes.find((attr) => attr.key === 'tool.call.id');
    return attribute?.value.string_value ?? '';
  }
  return child.attributes?.['tool.call.id'] as string;
};

const getToolNameFromSpan = (child: ModelTraceSpanNode): string => {
  if (Array.isArray(child.attributes)) {
    const attribute = child.attributes.find((attr) => attr.key === 'tool.name');
    return attribute?.value.string_value ?? '';
  }
  return child.attributes?.['tool.name'] as string;
};

export const synthesizeVoltAgentChatMessages = (
  inputs: any,
  outputs: any,
  children: ModelTraceSpanNode[],
): ModelTraceChatMessage[] | null => {
  const messages: ModelTraceChatMessage[] = [];

  const inputMessages = normalizeVoltAgentChatInput(inputs);

  if (!inputMessages || inputMessages.length === 0) {
    return null;
  }

  messages.push(...inputMessages);

  const toolSpans = children.filter((child) => {
    if (Array.isArray(child.attributes)) {
      const attribute = child.attributes.find((attr) => attr.key === 'span.type');
      return attribute?.value.string_value === 'tool';
    }
    return child.attributes?.['span.type'] === 'tool';
  });

  if (toolSpans.length > 0) {
    const toolCalls: ModelTraceToolCall[] = toolSpans.map((toolSpan) => {
      const toolCallId = getToolCallIdFromSpan(toolSpan);
      const toolName = getToolNameFromSpan(toolSpan);
      const toolInput = toolSpan.inputs;

      return {
        id: toolCallId,
        function: {
          name: toolName,
          arguments: isObject(toolInput) ? JSON.stringify(toolInput, null, 2) : String(toolInput ?? ''),
        },
      };
    });

    const assistantWithToolCalls = prettyPrintChatMessage({
      role: 'assistant',
      content: undefined,
      tool_calls: toolCalls,
    });

    if (assistantWithToolCalls) {
      messages.push(assistantWithToolCalls);
    }

    for (const toolSpan of toolSpans) {
      const toolCallId = getToolCallIdFromSpan(toolSpan);
      const toolOutput = toolSpan.outputs;
      const toolResultContent = isObject(toolOutput) ? JSON.stringify(toolOutput, null, 2) : String(toolOutput ?? '');

      const toolResultMessage = prettyPrintChatMessage({
        role: 'tool',
        content: toolResultContent,
        tool_call_id: toolCallId,
      });

      if (toolResultMessage) {
        messages.push(toolResultMessage);
      }
    }
  }

  if (isString(outputs) && outputs.trim().length > 0) {
    const finalAssistantMessage = prettyPrintChatMessage({
      role: 'assistant',
      content: outputs,
    });

    if (finalAssistantMessage) {
      messages.push(finalAssistantMessage);
    }
  }

  return messages.length > 0 ? messages : null;
};

import { has, isArray, isObject, isString, get } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceSpanNode, ModelTraceToolCall } from '../ModelTrace.types';
import { prettyPrintChatMessage, prettyPrintToolCall } from '../ModelTraceExplorer.utils';

/**
 * Strands Agent Message Format:
 *
 * Input format: [{"role": "user", "content": [{"text": "..."}]}]
 * Output format: [{"toolUse": {...}}] or [{"text": "..."}] or [{"reasoningContent": {...}}]
 */

type StrandsToolUse = {
  toolUseId: string;
  name: string;
  input: any;
};

type StrandsReasoningContent = {
  reasoningText: {
    text: string;
    signature?: string;
  };
};

const isStrandsToolUse = (obj: unknown): obj is StrandsToolUse => {
  if (!isObject(obj)) return false;
  return isString((obj as any).toolUseId) && isString((obj as any).name);
};

const isStrandsReasoningContent = (obj: unknown): obj is StrandsReasoningContent => {
  if (!isObject(obj)) return false;
  const typedObj = obj as any;
  return isObject(typedObj.reasoningText) && isString(typedObj.reasoningText.text);
};

/**
 * Extract text from Strands content format: [{"text": "..."}] or "..."
 */
const extractTextContent = (content: unknown): string | undefined => {
  if (isString(content)) return content;
  if (!isArray(content)) return undefined;

  const texts = content.filter((item) => isObject(item) && isString((item as any).text)).map((item) => (item as any).text);
  return texts.length > 0 ? texts.join('\n\n') : undefined;
};

/**
 * Parse output array into tool calls, text, and reasoning
 */
const parseOutputArray = (
  outputs: unknown[],
): { toolCalls: ModelTraceToolCall[]; textParts: string[]; reasoningParts: string[] } => {
  const toolCalls: ModelTraceToolCall[] = [];
  const textParts: string[] = [];
  const reasoningParts: string[] = [];

  for (const item of outputs) {
    if (!isObject(item)) continue;

    const reasoning = get(item, 'reasoningContent');
    if (isStrandsReasoningContent(reasoning)) {
      reasoningParts.push((reasoning as StrandsReasoningContent).reasoningText.text);
      continue;
    }

    const toolUse = get(item, 'toolUse');
    if (isStrandsToolUse(toolUse)) {
      const tu = toolUse as StrandsToolUse;
      toolCalls.push({
        id: tu.toolUseId,
        function: {
          name: tu.name,
          arguments: JSON.stringify(tu.input, null, 2),
        },
      });
    } else {
      const text = get(item, 'text');
      if (isString(text)) {
        textParts.push(text);
      }
    }
  }

  return { toolCalls, textParts, reasoningParts };
};

/**
 * Normalize Strands input messages
 */
export const normalizeStrandsChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!isArray(obj) || obj.length === 0) return null;

  const messages: ModelTraceChatMessage[] = [];

  for (const msg of obj) {
    if (!isObject(msg) || !has(msg, 'role') || !has(msg, 'content')) continue;

    const role = (msg as any).role;
    if (role !== 'user' && role !== 'system') continue;

    const text = extractTextContent((msg as any).content);
    const processed = prettyPrintChatMessage({ role, content: text });
    if (processed) messages.push(processed);
  }

  return messages.length > 0 ? messages : null;
};

/**
 * Normalize Strands output
 */
export const normalizeStrandsChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  // Plain string output
  if (isString(obj) && obj.trim()) {
    const msg = prettyPrintChatMessage({ role: 'assistant', content: obj });
    return msg ? [msg] : null;
  }

  if (!isArray(obj) || obj.length === 0) return null;

  const { toolCalls, textParts, reasoningParts } = parseOutputArray(obj);
  const reasoningText = reasoningParts.length > 0 ? reasoningParts.join('\n\n') : undefined;
  const contentText = textParts.length > 0 ? textParts.join('\n\n') : undefined;

  if (toolCalls.length > 0) {
    return [
      {
        role: 'assistant',
        content: contentText,
        tool_calls: toolCalls.map(prettyPrintToolCall),
        ...(reasoningText && { reasoning: reasoningText }),
      },
    ];
  }

  if (contentText) {
    const msg = prettyPrintChatMessage({ role: 'assistant', content: contentText });
    return msg ? [{ ...msg, ...(reasoningText && { reasoning: reasoningText }) }] : null;
  }

  if (reasoningText) {
    return [{ role: 'assistant', reasoning: reasoningText }];
  }

  return null;
};

/**
 * Recursively flatten all spans from the tree
 */
const flattenSpans = (children: ModelTraceSpanNode[]): ModelTraceSpanNode[] => {
  const result: ModelTraceSpanNode[] = [];
  for (const child of children) {
    result.push(child);
    if (child.children?.length) {
      result.push(...flattenSpans(child.children));
    }
  }
  return result;
};

/**
 * Synthesize chat messages from Strands agent spans
 */
export const synthesizeStrandsChatMessages = (
  inputs: any,
  outputs: any,
  children: ModelTraceSpanNode[],
): ModelTraceChatMessage[] | null => {
  const messages: ModelTraceChatMessage[] = [];

  // 1. Add user message from inputs
  const inputMessages = normalizeStrandsChatInput(inputs);
  if (inputMessages?.length) {
    messages.push(inputMessages[0]);
  }

  // 2. Find all CHAT_MODEL and TOOL spans
  const allSpans = flattenSpans(children);
  const chatSpans = allSpans.filter((s) => s.type === 'CHAT_MODEL');
  const toolSpans = allSpans.filter((s) => s.type === 'TOOL');

  // 3. Process CHAT_MODEL spans
  for (const chatSpan of chatSpans) {
    if (!isArray(chatSpan.outputs)) continue;

    const { toolCalls, textParts, reasoningParts } = parseOutputArray(chatSpan.outputs);
    const reasoningText = reasoningParts.length > 0 ? reasoningParts.join('\n\n') : undefined;
    const contentText = textParts.length > 0 ? textParts.join('\n\n') : undefined;

    if (toolCalls.length > 0) {
      // Assistant message with tool calls
      messages.push({
        role: 'assistant',
        content: contentText,
        tool_calls: toolCalls.map(prettyPrintToolCall),
        ...(reasoningText && { reasoning: reasoningText }),
      });

      // Add tool results
      for (const toolCall of toolCalls) {
        const toolSpan = toolSpans.find((ts) => get(ts.attributes, 'gen_ai.tool.call.id') === toolCall.id);
        if (!toolSpan) continue;

        const output = toolSpan.outputs;
        const resultContent = isArray(output) ? extractTextContent(output) ?? JSON.stringify(output) : String(output ?? '');

        const toolMsg = prettyPrintChatMessage({ role: 'tool', content: resultContent, tool_call_id: toolCall.id });
        if (toolMsg) messages.push(toolMsg);
      }
    } else if (reasoningText) {
      // CHAT_MODEL with reasoning but no tool calls
      messages.push({ role: 'assistant', content: contentText, reasoning: reasoningText });
    }
  }

  // 4. Add final response from root output (avoid duplicates)
  const hasAssistantWithContent = messages.some((m) => m.role === 'assistant' && m.content && !m.tool_calls);
  if (isString(outputs) && outputs.trim() && !hasAssistantWithContent) {
    const finalMsg = prettyPrintChatMessage({ role: 'assistant', content: outputs.trim() });
    if (finalMsg) messages.push(finalMsg);
  }

  return messages.length > 0 ? messages : null;
};

import { get, isArray, isObject, isString } from 'lodash';

import type { ModelTraceChatMessage, ModelTraceSpanNode, ModelTraceToolCall } from '../ModelTrace.types';
import { prettyPrintChatMessage } from '../ModelTraceExplorer.utils';

/**
 * Agno message structure.
 *
 *   System:    { role: "system", content: "You are helpful." }
 *   User:      { role: "user", content: "What is 2+2?" }
 *   Assistant: { role: "assistant", content: "4" }
 *   Assistant with tool calls: { role: "assistant", tool_calls: [{ id, type, function: { name, arguments } }] }
 *   Tool:      { role: "tool", content: "result", tool_call_id: "call_123", tool_name: "calculator" }
 *   Combined tool: { role: "tool", content: ["r1", "r2"], tool_calls: [{ tool_call_id, tool_name, content }] }
 */
type AgnoMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: string | string[] | null;
  tool_calls?: AgnoToolCall[] | AgnoToolResult[];
  tool_call_id?: string;
  tool_name?: string;
};

/**
 * Tool call from assistant message (requesting tool execution).
 *
 * Found in: assistant message's tool_calls array
 *   {
 *     id: "call_abc123",
 *     type: "function",
 *     function: {
 *       name: "get_stock_price",
 *       arguments: '{"symbol": "AAPL"}'
 *     }
 *   }
 */
type AgnoToolCall = {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
};

/**
 * Tool result in combined tool message.
 *
 * Found in: tool message's tool_calls array (when multiple tools executed)
 *   {
 *     tool_call_id: "call_abc123",
 *     tool_name: "get_stock_price",
 *     content: "$150.25"
 *   }
 */
type AgnoToolResult = {
  tool_call_id: string;
  tool_name: string;
  content: string;
};

/**
 * Check if object is a valid Agno message.
 *
 * Valid if: object has 'role' field with value 'system', 'user', 'assistant', or 'tool'
 *   { role: "user", content: "Hello" }
 *   { role: "assistant", tool_calls: [...] }
 *   { role: "tool", content: "result" }
 */
const isAgnoMessage = (obj: unknown): obj is AgnoMessage => {
  if (!isObject(obj)) return false;
  const role = get(obj, 'role');
  return isString(role) && ['system', 'user', 'assistant', 'tool'].includes(role);
};

/**
 * Check if object is an Agno tool call (from assistant).
 *
 * Valid if: object has 'function' field (indicating it's a tool call request)
 *   { id: "call_123", type: "function", function: { name: "calc", arguments: "{}" }
 */
const isAgnoToolCall = (tc: unknown): tc is AgnoToolCall => {
  if (!isObject(tc)) return false;
  return 'function' in (tc as Record<string, unknown>);
};

/**
 * Check if object is an Agno tool result.
 *
 * Valid if: object has both 'tool_call_id' AND 'tool_name' fields
 *   { tool_call_id: "call_123", tool_name: "get_weather", content: "72°F" }
 */
const isAgnoToolResult = (tc: unknown): tc is AgnoToolResult => {
  if (!isObject(tc)) return false;
  return 'tool_call_id' in (tc as Record<string, unknown>) && 'tool_name' in (tc as Record<string, unknown>);
};

/**
 * Convert Agno tool call to ModelTraceToolCall format.
 *
 * Input:  { id: "call_123", type: "function", function: { name: "calc", arguments: '{"x":1}' } }
 * Output: { id: "call_123", function: { name: "calc", arguments: '{"x":1}' } }
 */
const normalizeToolCall = (tc: AgnoToolCall): ModelTraceToolCall => ({
  id: tc.id,
  function: {
    name: tc.function.name,
    arguments: tc.function.arguments,
  },
});

/**
 * Normalize a tool message to ModelTraceChatMessage array.
 *
 * Handles three formats:
 *
 * 1. Combined tool message with tool_calls array (preferred):
 *    Input:  { role: "tool", tool_calls: [
 *              { tool_call_id: "call_1", tool_name: "weather", content: "72°F" },
 *              { tool_call_id: "call_2", tool_name: "stock", content: "$150" }
 *            ] }
 *    Output: [
 *              { role: "tool", content: "72°F", tool_call_id: "call_1", name: "weather" },
 *              { role: "tool", content: "$150", tool_call_id: "call_2", name: "stock" }
 *            ]
 *
 * 2. Content as array of strings:
 *    Input:  { role: "tool", content: ["72°F", "$150"] }
 *    Output: [{ role: "tool", content: "72°F" }, { role: "tool", content: "$150" }]
 *
 * 3. Single string content:
 *    Input:  { role: "tool", content: "72°F", tool_call_id: "call_1" }
 *    Output: [{ role: "tool", content: "72°F", tool_call_id: "call_1" }]
 */
const normalizeToolMessage = (msg: AgnoMessage): ModelTraceChatMessage[] => {
  const results: ModelTraceChatMessage[] = [];

  // Priority 1: Extract individual tool results from tool_calls array
  // Format: { role: "tool", tool_calls: [{ tool_call_id, tool_name, content }, ...] }
  const toolCalls = msg.tool_calls;
  if (isArray(toolCalls)) {
    const toolResults = toolCalls.filter(isAgnoToolResult);
    for (const tc of toolResults) {
      const m = prettyPrintChatMessage({
        role: 'tool',
        content: tc.content,
        tool_call_id: tc.tool_call_id,
        name: tc.tool_name,
      });
      if (m) results.push(m);
    }
  }

  // Priority 2: Fallback - use content array
  // Format: { role: "tool", content: ["result1", "result2"] }
  if (results.length === 0 && isArray(msg.content)) {
    const stringContents = msg.content.filter(isString);
    for (const c of stringContents) {
      const m = prettyPrintChatMessage({ role: 'tool', content: c });
      if (m) results.push(m);
    }
  }

  // Priority 3: Fallback - single string content
  // Format: { role: "tool", content: "result", tool_call_id: "call_1" }
  if (results.length === 0 && isString(msg.content)) {
    const m = prettyPrintChatMessage({
      role: 'tool',
      content: msg.content,
      ...(msg.tool_call_id && { tool_call_id: msg.tool_call_id }),
    });
    if (m) results.push(m);
  }

  return results;
};

/**
 * Normalize any Agno message to ModelTraceChatMessage array.
 *
 * Routes to appropriate handler based on role:
 *
 * Tool message (role: "tool"):
 *   Delegates to normalizeToolMessage()
 *
 * Assistant message (role: "assistant"):
 *   Input:  { role: "assistant", content: "Hello!", tool_calls: [{ id, function: {...} }] }
 *   Output: [{ role: "assistant", content: "Hello!", tool_calls: [...] }]
 *
 * System/User message (role: "system" | "user"):
 *   Input:  { role: "user", content: "What is 2+2?" }
 *   Output: [{ role: "user", content: "What is 2+2?" }]
 *
 *   Input:  { role: "user", content: ["line1", "line2"] }  // array content
 *   Output: [{ role: "user", content: "line1\nline2" }]    // joined with newlines
 */
const normalizeMessage = (msg: AgnoMessage): ModelTraceChatMessage[] => {
  // Tool messages have special handling for combined results
  if (msg.role === 'tool') {
    return normalizeToolMessage(msg);
  }

  // Assistant message - may have tool_calls
  // Format: { role: "assistant", content?: "...", tool_calls?: [{ id, type, function }] }
  if (msg.role === 'assistant') {
    const toolCalls = isArray(msg.tool_calls) ? msg.tool_calls.filter(isAgnoToolCall).map(normalizeToolCall) : [];
    const content = isString(msg.content) ? msg.content : undefined;
    const m = prettyPrintChatMessage({
      role: 'assistant',
      content,
      ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
    });
    return m ? [m] : [];
  }

  // System or user message
  // Format: { role: "system"|"user", content: "..." } or { role: "...", content: ["line1", "line2"] }
  const content = isString(msg.content) ? msg.content : isArray(msg.content) ? msg.content.join('\n') : undefined;
  const m = prettyPrintChatMessage({ role: msg.role, content });
  return m ? [m] : [];
};

/**
 * Normalize an array of Agno messages.
 *
 * Input:  [{ role: "system", content: "..." }, { role: "user", content: "..." }]
 * Output: [{ role: "system", content: "..." }, { role: "user", content: "..." }]
 */
const normalizeMessagesArray = (messages: unknown[]): ModelTraceChatMessage[] => {
  const result: ModelTraceChatMessage[] = [];
  messages.forEach((m) => {
    if (isAgnoMessage(m)) {
      result.push(...normalizeMessage(m));
    }
  });
  return result;
};

/**
 * Parse Agno INPUT data.
 *
 * Handles two formats:
 *
 * 1. LLM span input (object with messages array):
 *    Input:  { messages: [{ role: "system", content: "..." }, { role: "user", content: "..." }] }
 *    Output: [{ role: "system", content: "..." }, { role: "user", content: "..." }]
 *
 * 2. AGENT span input (plain string, NOT JSON):
 *    Input:  "What is the stock price of Apple?"
 *    Output: [{ role: "user", content: "What is the stock price of Apple?" }]
 *
 * Does NOT handle:
 *    - JSON strings starting with '[' or '{' (those go to normalizeAgnoChatOutput)
 *    - null/undefined values
 */
export const normalizeAgnoChatInput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!obj) return null;

  // Format 1: LLM span input - object with messages array
  // Example: { messages: [{ role: "user", content: "Hello" }] }
  if (isObject(obj)) {
    const messagesArr = get(obj, 'messages');
    if (isArray(messagesArr)) {
      const result = normalizeMessagesArray(messagesArr);
      return result.length > 0 ? result : null;
    }
  }

  // Format 2: AGENT span input - plain string (NOT JSON)
  // Example: "What is the stock price of Apple?"
  // Excludes: '[{"role":"assistant",...}]' or '{"key":"value"}' (these are JSON, handled elsewhere)
  if (isString(obj) && !obj.startsWith('[') && !obj.startsWith('{')) {
    const msg = prettyPrintChatMessage({ role: 'user', content: obj });
    return msg ? [msg] : null;
  }

  return null;
};

/**
 * Parse Agno OUTPUT data.
 *
 * Handles two formats:
 *
 * 1. LLM span output (JSON string of message array):
 *    Input:  '[{"role":"assistant","content":"Hello!"}]'
 *    Output: [{ role: "assistant", content: "Hello!" }]
 *
 *    Input:  '[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"calc","arguments":"{}"}}]}]'
 *    Output: [{ role: "assistant", tool_calls: [{ id: "call_1", function: { name: "calc", arguments: "{}" } }] }]
 *
 * 2. AGENT span output (plain markdown string):
 *    Input:  "## Stock Price\n\nApple is trading at $150."
 *    Output: [{ role: "assistant", content: "## Stock Price\n\nApple is trading at $150." }]
 *
 */
export const normalizeAgnoChatOutput = (obj: unknown): ModelTraceChatMessage[] | null => {
  if (!obj) return null;

  if (isString(obj)) {
    // Format 1: LLM output - JSON string starting with '[' (array of messages)
    // Example: '[{"role":"assistant","content":"The answer is 42."}]'
    if (obj.startsWith('[')) {
      try {
        const parsed = JSON.parse(obj);
        if (isArray(parsed)) {
          const result = normalizeMessagesArray(parsed);
          return result.length > 0 ? result : null;
        }
      } catch {
        // Not valid JSON, fall through to plain string handling
      }
    }

    // Format 2: AGENT output - plain string (markdown response)
    // Example: "## Results\n\nThe stock price is $150."
    const msg = prettyPrintChatMessage({ role: 'assistant', content: obj });
    return msg ? [msg] : null;
  }

  return null;
};

/**
 * Extract system prompt from the first LLM child span's inputs.
 *
 * LLM span inputs have this structure:
 *   { messages: [{ role: "system", content: "..." }, { role: "user", content: "..." }, ...] }
 *
 * We extract the first system message to show in the AGENT ChatUI.
 */
const extractSystemPromptFromChildren = (children: ModelTraceSpanNode[]): ModelTraceChatMessage | null => {
  for (const child of children) {
    const spanType = child.type || child.attributes?.['mlflow.spanType'];
    if (spanType !== 'LLM') continue;

    // LLM span inputs: { messages: [...] }
    const llmInputs = child.inputs;
    if (!isObject(llmInputs)) continue;

    const messages = get(llmInputs, 'messages');
    if (!isArray(messages)) continue;

    // Find the first system message
    for (const msg of messages as unknown[]) {
      if (isObject(msg) && get(msg, 'role') === 'system') {
        const content = get(msg, 'content') as unknown;
        if (isString(content) && content.trim()) {
          return prettyPrintChatMessage({ role: 'system', content: content.trim() });
        }
      }
    }
    // Only check the first LLM span
    break;
  }
  return null;
};

/**
 * Synthesize chat messages for AGENT spans from child LLM/TOOL spans.
 *
 * AGENT spans only have simple string input/output. The full conversation
 * with tool calls is in child LLM spans. This function aggregates them.
 *
 * Input structure:
 *   inputs:   "What is the stock price?" (plain string - user query)
 *   outputs:  "Apple is $150." (plain string - final answer)
 *   children: [
 *     { type: "LLM", inputs: { messages: [{ role: "system", content: "..." }, ...] }, outputs: '[{"role":"assistant","tool_calls":[...]}]' },
 *     { type: "TOOL", attributes: { "tool.name": "get_stock" }, outputs: "$150" },
 *     { type: "LLM", outputs: '[{"role":"assistant","content":"Apple is $150."}]' }
 *   ]
 *
 * Output:
 *   [
 *     { role: "system", content: "You are a helpful assistant..." },
 *     { role: "user", content: "What is the stock price?" },
 *     { role: "assistant", tool_calls: [...] },
 *     { role: "tool", content: "$150", name: "get_stock" },
 *     { role: "assistant", content: "Apple is $150." }
 *   ]
 */
export const synthesizeAgnoChatMessages = (
  inputs: unknown,
  outputs: unknown,
  children: ModelTraceSpanNode[],
): ModelTraceChatMessage[] | null => {
  const messages: ModelTraceChatMessage[] = [];

  // Extract system prompt from first LLM child span
  const systemPrompt = extractSystemPromptFromChildren(children);
  if (systemPrompt) {
    messages.push(systemPrompt);
  }

  // Add user input
  // Format: plain string "What is the stock price?"
  const inputMessages = normalizeAgnoChatInput(inputs);
  if (!inputMessages?.length) return null;
  messages.push(...inputMessages);

  // Process children IN ORDER (chronological execution order)
  // We must NOT group by type, as that breaks the conversation flow
  for (const child of children) {
    const spanType = child.type || child.attributes?.['mlflow.spanType'];

    if (spanType === 'LLM') {
      // LLM span output: '[{"role":"assistant","content":"..."}]' or '[{"role":"assistant","tool_calls":[...]}]'
      const llmMessages = normalizeAgnoChatOutput(child.outputs);
      const assistantMessages = llmMessages?.filter((msg) => msg.role === 'assistant') ?? [];
      messages.push(...assistantMessages);
    } else if (spanType === 'TOOL') {
      // TOOL span: { attributes: { "tool.name": "get_weather" }, outputs: "72°F" }
      const toolName = (child.attributes?.['tool.name'] as string) || String(child.title || child.key);
      const toolOutput = isObject(child.outputs) ? JSON.stringify(child.outputs, null, 2) : String(child.outputs || '');
      const msg = prettyPrintChatMessage({
        role: 'tool',
        content: toolOutput,
        name: toolName,
      });
      if (msg) messages.push(msg);
    }
  }

  return messages.length > 0 ? messages : null;
};

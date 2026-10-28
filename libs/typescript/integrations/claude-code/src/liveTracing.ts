/**
 * Live trace construction for the Claude Agent SDK message stream.
 *
 * Mirrors the span shape produced by processTranscript (CLI Stop-hook path)
 * but emits spans incrementally as SDKMessages arrive so that long-running
 * agent sessions show progress and partial traces survive crashes.
 *
 * Span tree (matches CLI integration):
 *   claude_code_conversation (AGENT)
 *   ├── llm (LLM)              per assistant response, including tool-only responses
 *   ├── tool_<name> (TOOL)     per tool_use block at root
 *   │   └── subagent_<type> (AGENT)   lazily created for Task/Agent tool calls
 *   │       ├── llm
 *   │       └── tool_<name>
 *   └── ...
 *
 * Sub-agent attribution is driven by SDKMessage.parent_tool_use_id (set by
 * the SDK when forwardSubagentText: true is passed on query() options).
 */

import {
  startSpan,
  flushTraces,
  SpanType,
  SpanAttributeKey,
  SpanStatusCode,
  TraceMetadataKey,
  InMemoryTraceManager,
  type LiveSpan,
} from '@mlflow/core';

import type { ContentBlock, MessageContent, TokenUsage } from './types.js';
import {
  MAX_PREVIEW_LENGTH,
  METADATA_KEY_CLAUDE_CODE_VERSION,
  METADATA_KEY_PERMISSION_MODE,
  METADATA_KEY_WORKING_DIRECTORY,
  buildUsageDict,
  extractContentAndTools,
  getCurrentUser,
  mergeAssistantMessage,
  sanitizeForSpan,
} from './_internal.js';

// Tool names that invoke a sub-agent. `'Agent'` is what the Claude Agent SDK
// emits as of 0.2.x (confirmed via live stream inspection); `'Task'` is the
// legacy name still used by the Claude Code CLI transcript. Both surfaces
// land in this code path now that SDK and CLI traces share their builders.
const SUBAGENT_TOOL_NAMES = new Set(['Task', 'Agent']);

type ConversationKey = string | null;

interface InputMessage {
  role: 'user' | 'assistant';
  content: string | ContentBlock[];
}

interface PendingLlmMessage {
  message: MessageContent;
  span: LiveSpan;
  conversationEntry: InputMessage;
  endTimeNs: number;
}

/**
 * Minimal structural types for the SDKMessage subset we consume. The wrapper
 * narrows real SDKMessage values to these shapes before dispatching, so the
 * SDK package itself is only a peer dependency (and only required for types).
 */
export interface SDKSystemInit {
  type: 'system';
  subtype: 'init';
  session_id?: string;
  model?: string;
  cwd?: string;
  permissionMode?: string;
  claude_code_version?: string;
}

export interface SDKAssistantLike {
  type: 'assistant';
  message: MessageContent & { usage?: TokenUsage };
  parent_tool_use_id?: string | null;
}

export interface SDKUserLike {
  type: 'user';
  message: { role: 'user'; content: string | ContentBlock[] };
  parent_tool_use_id?: string | null;
}

export interface SDKResultLike {
  type: 'result';
  /** 'success' on normal completion; any other value is an SDK-defined error subtype. */
  subtype: string;
  duration_ms?: number;
  total_cost_usd?: number;
  usage?: TokenUsage;
  result?: string;
}

export class LiveTracingContext {
  private rootSpan: LiveSpan;
  private openToolSpans = new Map<string, LiveSpan>();
  private subagentSpans = new Map<string, LiveSpan>();
  private conversations = new Map<ConversationKey, InputMessage[]>();
  private lastAssistantText = new Map<ConversationKey, string>();
  // Split content records update the same span; close it at the final record's timestamp.
  private pendingLlmMessages = new Map<ConversationKey, PendingLlmMessage>();
  private emittedToolIds = new Set<string>();
  private sessionId: string | undefined;
  private model: string | undefined;
  private permissionMode: string | undefined;
  private claudeCodeVersion: string | undefined;
  private ended = false;
  private readonly spanOptions: unknown;

  constructor(prompt: string, options?: Record<string, unknown>) {
    this.spanOptions = sanitizeOptions(options);
    this.rootSpan = startSpan({
      name: 'claude_code_conversation',
      spanType: SpanType.AGENT,
      inputs: { prompt, options: this.spanOptions },
    });
    this.conversations.set(null, [{ role: 'user', content: prompt }]);
  }

  /**
   * Append captured prompt text (used when the caller passes an AsyncIterable
   * prompt to query() — content arrives over time, not as a single string).
   * Each call refreshes the root span's `inputs.prompt` so users see the most
   * complete prompt available in the trace UI as more chunks arrive.
   */
  appendPromptText(text: string): void {
    const conversation = this.getConversation(null);
    const first = conversation[0];
    if (first?.role === 'user' && typeof first.content === 'string') {
      first.content = first.content ? `${first.content}\n${text}` : text;
    } else {
      conversation.unshift({ role: 'user', content: text });
    }
    this.rootSpan.setInputs({ prompt: conversation[0].content, options: this.spanOptions });
  }

  /** Update root span metadata with session-level info from the init message. */
  onSystemInit(msg: SDKSystemInit): void {
    this.sessionId = msg.session_id ?? this.sessionId;
    this.model = msg.model ?? this.model;
    this.permissionMode = msg.permissionMode ?? this.permissionMode;
    this.claudeCodeVersion = msg.claude_code_version ?? this.claudeCodeVersion;
  }

  /**
   * Emit an LLM span for each response and open TOOL spans
   * for each tool_use block. Sub-agent routing is keyed on parent_tool_use_id.
   */
  onAssistantMessage(msg: SDKAssistantLike): void {
    const parentKey: ConversationKey = msg.parent_tool_use_id ?? null;
    const parentSpan = this.resolveParent(parentKey);
    const conversation = this.getConversation(parentKey);
    let pending = this.pendingLlmMessages.get(parentKey);
    const timestampNs = (performance.timeOrigin + performance.now()) * 1e6;
    if (pending && msg.message.id && pending.message.id === msg.message.id) {
      pending.message = mergeAssistantMessage(pending.message, msg.message);
      pending.conversationEntry.content = pending.message.content;
      pending.endTimeNs = timestampNs;
    } else {
      this.closeLlmMessage(parentKey);
      const model = msg.message.model ?? this.model ?? 'unknown';
      const llmSpan = startSpan({
        name: 'llm',
        spanType: SpanType.LLM,
        parent: parentSpan,
        startTimeNs: timestampNs,
        inputs: { model, messages: conversation.slice() },
        attributes: {
          model,
          'mlflow.llm.model': model,
          [SpanAttributeKey.MESSAGE_FORMAT]: 'anthropic',
        },
      });
      const conversationEntry: InputMessage = { role: 'assistant', content: msg.message.content };
      conversation.push(conversationEntry);
      pending = { message: msg.message, span: llmSpan, conversationEntry, endTimeNs: timestampNs };
      this.pendingLlmMessages.set(parentKey, pending);
    }
    if (pending.message.usage) {
      pending.span.setAttribute(
        SpanAttributeKey.TOKEN_USAGE,
        buildUsageDict(pending.message.usage),
      );
    }
    pending.span.setOutputs({
      type: 'message',
      role: 'assistant',
      content: pending.message.content,
    });
    const [textContent] = extractContentAndTools(pending.message.content);
    if (textContent.trim()) {
      this.lastAssistantText.set(parentKey, textContent);
    }

    const [, toolUses] = extractContentAndTools(msg.message.content);
    for (const toolUse of toolUses) {
      if (this.emittedToolIds.has(toolUse.id)) {
        continue;
      }
      this.emittedToolIds.add(toolUse.id);
      const toolSpan = startSpan({
        name: `tool_${toolUse.name}`,
        spanType: SpanType.TOOL,
        parent: parentSpan,
        startTimeNs: timestampNs,
        inputs: toolUse.input ?? {},
        attributes: { tool_name: toolUse.name, tool_id: toolUse.id },
      });
      this.openToolSpans.set(toolUse.id, toolSpan);

      // Lazily create the sub-agent wrapper span when we see a Task/Agent tool
      // call. Sub-agent inner messages arrive with parent_tool_use_id === toolUse.id,
      // and resolveParent() will route them here.
      if (SUBAGENT_TOOL_NAMES.has(toolUse.name)) {
        const subagentType = (toolUse.input?.subagent_type as string | undefined) ?? 'agent';
        const subagentSpan = startSpan({
          name: `subagent_${subagentType}`,
          spanType: SpanType.AGENT,
          parent: toolSpan,
          startTimeNs: timestampNs,
          inputs: {
            prompt: toolUse.input?.prompt,
            description: toolUse.input?.description,
            subagent_type: subagentType,
          },
          attributes: { subagent_type: subagentType },
        });
        this.subagentSpans.set(toolUse.id, subagentSpan);
      }
    }
  }

  /** Close any TOOL spans whose tool_use_id matches a tool_result block. */
  onUserMessage(msg: SDKUserLike): void {
    const parentKey: ConversationKey = msg.parent_tool_use_id ?? null;
    const conversation = this.getConversation(parentKey);
    conversation.push({ role: 'user', content: msg.message.content });

    const content = msg.message.content;
    if (!Array.isArray(content)) {
      return;
    }

    for (const block of content) {
      if (
        typeof block !== 'object' ||
        block == null ||
        !('type' in block) ||
        block.type !== 'tool_result'
      ) {
        continue;
      }
      const toolResult = block as {
        type: 'tool_result';
        tool_use_id?: string;
        content?: unknown;
        is_error?: boolean;
      };
      const toolUseId = toolResult.tool_use_id;
      if (!toolUseId) {
        continue;
      }

      // If this tool result closes a Task/Agent call, finalize its sub-agent
      // wrapper span first so it nests under the TOOL span correctly.
      const subagentSpan = this.subagentSpans.get(toolUseId);
      if (subagentSpan) {
        this.closeLlmMessage(toolUseId);
        const lastText = this.lastAssistantText.get(toolUseId);
        if (lastText) {
          subagentSpan.setOutputs({ response: lastText });
        }
        subagentSpan.end();
        this.subagentSpans.delete(toolUseId);
        this.conversations.delete(toolUseId);
        this.lastAssistantText.delete(toolUseId);
      }

      const toolSpan = this.openToolSpans.get(toolUseId);
      if (!toolSpan) {
        continue;
      }

      toolSpan.setOutputs({ result: toolResult.content ?? '' });
      if (toolResult.is_error) {
        const errorText =
          typeof toolResult.content === 'string'
            ? toolResult.content
            : JSON.stringify(toolResult.content);
        toolSpan.setStatus(SpanStatusCode.ERROR, errorText || 'Tool execution failed');
      }
      toolSpan.end();
      this.openToolSpans.delete(toolUseId);
    }
  }

  /** Record aggregate usage and final result text from the SDK's result message. */
  onResultMessage(msg: SDKResultLike): void {
    this.closeLlmMessage(null);
    if (msg.usage) {
      this.rootSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, buildUsageDict(msg.usage));
    }
    if (msg.subtype !== 'success') {
      this.rootSpan.setStatus(SpanStatusCode.ERROR, msg.subtype);
    }
    const finalText = msg.result ?? this.lastAssistantText.get(null);
    if (finalText) {
      this.lastAssistantText.set(null, finalText);
    }
  }

  /** Close all open spans with ERROR status on stream throw or interruption. */
  async finalizeError(error: string): Promise<void> {
    if (this.ended) {
      return;
    }
    for (const span of this.openToolSpans.values()) {
      span.setStatus(SpanStatusCode.ERROR, error);
      span.end();
    }
    this.openToolSpans.clear();
    for (const span of this.subagentSpans.values()) {
      span.setStatus(SpanStatusCode.ERROR, error);
      span.end();
    }
    this.subagentSpans.clear();
    this.rootSpan.setStatus(SpanStatusCode.ERROR, error);
    await this.finalize();
  }

  /** Close the root span and flush traces to the backend; safe to call multiple times. */
  async finalize(): Promise<void> {
    if (this.ended) {
      return;
    }
    this.ended = true;

    for (const key of this.pendingLlmMessages.keys()) {
      this.closeLlmMessage(key);
    }

    // Defensive: close anything still open (shouldn't happen in normal completion).
    for (const span of this.openToolSpans.values()) {
      span.end();
    }
    this.openToolSpans.clear();
    for (const span of this.subagentSpans.values()) {
      span.end();
    }
    this.subagentSpans.clear();

    this.applyTraceMetadata();

    const finalResponse = this.lastAssistantText.get(null);
    const outputs: Record<string, unknown> = { status: 'completed' };
    if (finalResponse) {
      outputs.response = finalResponse;
    }
    this.rootSpan.setOutputs(outputs);
    this.rootSpan.end();

    try {
      await flushTraces();
    } catch (err) {
      console.error('[mlflow] Failed to flush traces:', err);
    }
  }

  private getConversation(key: ConversationKey): InputMessage[] {
    let conv = this.conversations.get(key);
    if (!conv) {
      conv = [];
      this.conversations.set(key, conv);
    }
    return conv;
  }

  private closeLlmMessage(key: ConversationKey): void {
    const pending = this.pendingLlmMessages.get(key);
    if (pending) {
      pending.span.end({ endTimeNs: pending.endTimeNs });
      this.pendingLlmMessages.delete(key);
    }
  }

  private resolveParent(parentKey: ConversationKey): LiveSpan {
    if (parentKey == null) {
      return this.rootSpan;
    }
    const subagent = this.subagentSpans.get(parentKey);
    if (subagent) {
      return subagent;
    }
    // Sub-agent wrapper hasn't been created yet — fall back to root. This only
    // happens if a sub-agent message arrives before its Task tool_use, which
    // the SDK never does in practice. We avoid silently dropping the span.
    return this.rootSpan;
  }

  private applyTraceMetadata(): void {
    try {
      const traceManager = InMemoryTraceManager.getInstance();
      const trace = traceManager.getTrace(this.rootSpan.traceId);
      if (!trace) {
        return;
      }

      const userPrompt = this.conversations.get(null)?.[0]?.content;
      if (typeof userPrompt === 'string') {
        trace.info.requestPreview = userPrompt.slice(0, MAX_PREVIEW_LENGTH);
      }
      const finalResponse = this.lastAssistantText.get(null);
      if (finalResponse) {
        trace.info.responsePreview = finalResponse.slice(0, MAX_PREVIEW_LENGTH);
      }

      const metadata: Record<string, string> = { ...trace.info.traceMetadata };
      const runId = process.env.MLFLOW_RUN_ID;
      if (runId) {
        metadata[TraceMetadataKey.SOURCE_RUN] = runId;
      }
      if (this.sessionId) {
        metadata[TraceMetadataKey.TRACE_SESSION] = this.sessionId;
      }
      const user = getCurrentUser();
      if (user) {
        metadata[TraceMetadataKey.TRACE_USER] = user;
      }
      metadata[METADATA_KEY_WORKING_DIRECTORY] = process.cwd();
      if (this.permissionMode) {
        metadata[METADATA_KEY_PERMISSION_MODE] = this.permissionMode;
      }
      if (this.claudeCodeVersion) {
        metadata[METADATA_KEY_CLAUDE_CODE_VERSION] = this.claudeCodeVersion;
      }
      trace.info.traceMetadata = metadata;
    } catch (err) {
      console.error('[mlflow] Failed to apply trace metadata:', err);
    }
  }
}

/**
 * Strip callbacks (functions are not serialisable and may close over large
 * objects) before recording options on the root span's inputs. The Agent SDK
 * options can include functions at several depths — `hooks[].hooks[]`,
 * `canUseTool`, MCP server entries — so we recursively drop any function
 * value rather than enumerating known fields.
 */
function sanitizeOptions(options: Record<string, unknown> | undefined): unknown {
  if (!options) {
    return undefined;
  }
  return sanitizeForSpan(options);
}

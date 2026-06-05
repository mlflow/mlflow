/**
 * Live trace construction for the Claude Agent SDK message stream.
 *
 * Mirrors the span shape produced by processTranscript (CLI Stop-hook path)
 * but emits spans incrementally as SDKMessages arrive so that long-running
 * agent sessions show progress and partial traces survive crashes.
 *
 * Span tree (matches CLI integration):
 *   claude_code_conversation (AGENT)
 *   ├── llm (LLM)              per assistant turn with text/thinking
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

/**
 * Minimal duck-type for SDK user messages. We deliberately under-declare vs
 * the full SDKUserMessage (sdk.d.ts:3657-3676) — only the fields the live
 * tracing path reads. Fields added here must reflect real wire properties.
 */
export interface SDKUserLike {
  type: 'user';
  message: { role: 'user'; content: string | ContentBlock[] };
  parent_tool_use_id?: string | null;
  tool_use_result?: { commandName?: string } | null;
  origin?: { kind?: string }; // sdk.d.ts:3664; values from SDKMessageOrigin sdk.d.ts:3138-3151
  isSynthetic?: boolean; // sdk.d.ts:3661
  shouldQuery?: boolean; // sdk.d.ts:3669; false = don't trigger an assistant turn
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
  private sessionId: string | undefined;
  private model: string | undefined;
  private permissionMode: string | undefined;
  private claudeCodeVersion: string | undefined;
  private activeSkillName: string | undefined;
  // Tracks whether the previous user message carried a Skill tool_use_result so
  // we can detect skill content injections (which always follow such a result
  // and look structurally identical to a real user prompt). Mirrors the
  // look-back used by findLastUserMessageIndex in transcript.ts:109-121.
  private prevUserHadCommandName = false;
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
    // Reset both skill-scope state fields together: clearing activeSkillName
    // without also clearing prevUserHadCommandName would leave the next user
    // message classified as a skill body injection.
    this.activeSkillName = undefined;
    this.prevUserHadCommandName = false;
  }

  /**
   * Emit an LLM span (if the message has text/thinking) and open TOOL spans
   * for each tool_use block. Sub-agent routing is keyed on parent_tool_use_id.
   */
  onAssistantMessage(msg: SDKAssistantLike): void {
    const parentKey: ConversationKey = msg.parent_tool_use_id ?? null;
    const parentSpan = this.resolveParent(parentKey);
    const content = msg.message.content;
    const [textContent, toolUses] = extractContentAndTools(content);

    const conversation = this.getConversation(parentKey);
    const priorMessages = conversation.slice();
    conversation.push({ role: 'assistant', content });

    const model = msg.message.model ?? this.model ?? 'unknown';
    if (textContent.trim() || hasThinking(content)) {
      const llmSpan = startSpan({
        name: 'llm',
        spanType: SpanType.LLM,
        parent: parentSpan,
        inputs: { model, messages: priorMessages },
        attributes: {
          model,
          'mlflow.llm.model': model,
          [SpanAttributeKey.MESSAGE_FORMAT]: 'anthropic',
          ...(this.activeSkillName
            ? { [SpanAttributeKey.SKILL_NAME]: this.activeSkillName }
            : {}),
        },
      });
      if (msg.message.usage) {
        llmSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, buildUsageDict(msg.message.usage));
      }
      llmSpan.setOutputs({ type: 'message', role: 'assistant', content });
      llmSpan.end();

      if (textContent.trim()) {
        this.lastAssistantText.set(parentKey, textContent);
      }
    }

    for (const toolUse of toolUses) {
      const toolSpan = startSpan({
        name: `tool_${toolUse.name}`,
        spanType: SpanType.TOOL,
        parent: parentSpan,
        inputs: toolUse.input ?? {},
        attributes: {
          tool_name: toolUse.name,
          tool_id: toolUse.id,
          ...(this.activeSkillName
            ? { [SpanAttributeKey.SKILL_NAME]: this.activeSkillName }
            : {}),
        },
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
          inputs: {
            prompt: toolUse.input?.prompt,
            description: toolUse.input?.description,
            subagent_type: subagentType,
          },
          attributes: {
            subagent_type: subagentType,
            ...(this.activeSkillName
              ? { [SpanAttributeKey.SKILL_NAME]: this.activeSkillName }
              : {}),
          },
        });
        this.subagentSpans.set(toolUse.id, subagentSpan);
      }
    }
  }

  /**
   * Returns true if `msg` is a fresh, human-originated prompt — as opposed to
   * a tool-result echo, sub-agent inner turn, skill content injection, or any
   * other SDK-emitted non-prompt user-role message.
   *
   * Used to decide whether to clear `activeSkillName` so that work triggered
   * by a *new* user prompt isn't mis-attributed to a previously active skill.
   * Relevant to the AsyncIterable multi-prompt case: `query()` accepts
   * `AsyncIterable<SDKUserMessage>`, which streams multiple prompts through
   * a single LiveTracingContext. See
   * https://docs.claude.com/en/api/agent-sdk/python.
   *
   * The check is layered: strong SDK-stamped signal first, then fallbacks for
   * older SDK versions and programmatic emitters that may not set it.
   *
   * Wire-level field references from
   * @anthropic-ai/claude-agent-sdk/sdk.d.ts (v0.2.0):
   *   - SDKMessageOrigin: sdk.d.ts:3138-3151
   *   - SDKUserMessage:   sdk.d.ts:3657-3676
   *
   * NOTE: we intentionally do NOT check for a leading `tool_result` block in
   * `content` (as transcript.ts:130-137 does for the on-disk transcript)
   * because in the live SDK stream tool-result echoes always carry
   * `tool_use_result != null` (caught below). Verified by reading the SDK
   * parser at @anthropic-ai/claude-agent-sdk/_internal/message_parser.py.
   */
  private isRealUserPrompt(msg: SDKUserLike): boolean {
    // Strong signal: SDK-stamped origin (sdk.d.ts:3664).
    if (msg.origin?.kind === 'human') {return true;}
    if (msg.origin?.kind != null) {return false;}

    // SDK metadata that explicitly says "not a real prompt".
    if (msg.isSynthetic) {return false;}
    if (msg.shouldQuery === false) {return false;}

    // Fallback heuristics for older emitters with no `origin`.
    if (msg.tool_use_result != null) {return false;}
    if (msg.parent_tool_use_id != null) {return false;}

    // Skill content injection: Claude Code injects the skill body as a `user`
    // message immediately after a Skill tool_result. Structurally identical
    // to a real prompt — only position distinguishes it. Mirrors the
    // transcript-path heuristic at transcript.ts:109-121 and
    // mlflow/claude_code/tracing.py:245-254.
    if (this.prevUserHadCommandName) {return false;}

    // Empty / whitespace-only string content is never a prompt. Also treat
    // an empty content array as empty (no text or blocks to act on).
    const content = msg.message.content;
    if (typeof content === 'string' && !content.trim()) {return false;}
    if (Array.isArray(content) && content.length === 0) {return false;}

    return true;
  }

  /** Close any TOOL spans whose tool_use_id matches a tool_result block. */
  onUserMessage(msg: SDKUserLike): void {
    // A new human prompt arriving mid-stream (e.g. AsyncIterable prompts to
    // query()) means any previously active skill scope is over. Read
    // hadCommandName from this message up-front, but defer mutating
    // prevUserHadCommandName until after the tool-result loop runs so an
    // exception there can't leave the look-back state inconsistent.
    const hadCommandName = !!msg.tool_use_result?.commandName;
    try {
      if (this.isRealUserPrompt(msg)) {
        this.activeSkillName = undefined;
      }

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
      // Always stamp the Skill TOOL span with its own commandName for
      // identification (failed Skills are still attributable).
      // Propagation:
      //   - success: activeSkillName = commandName (child spans inherit)
      //   - failure: activeSkillName = undefined (prior skill must not leak
      //     into recovery spans, and the failed skill never injected its body
      //     so it shouldn't propagate either)
      if (msg.tool_use_result?.commandName) {
        toolSpan.setAttribute(SpanAttributeKey.SKILL_NAME, msg.tool_use_result.commandName);
        this.activeSkillName = toolResult.is_error
          ? undefined
          : msg.tool_use_result.commandName;
      }
      toolSpan.end();
      this.openToolSpans.delete(toolUseId);
      }
    } finally {
      this.prevUserHadCommandName = hadCommandName;
    }
  }

  /** Record aggregate usage and final result text from the SDK's result message. */
  onResultMessage(msg: SDKResultLike): void {
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
      if (this.sessionId) {
        metadata[TraceMetadataKey.TRACE_SESSION] = this.sessionId;
      }
      const user = process.env.USER;
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

function hasThinking(content: string | ContentBlock[]): boolean {
  return (
    Array.isArray(content) &&
    content.some((b) => typeof b === 'object' && b != null && 'type' in b && b.type === 'thinking')
  );
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

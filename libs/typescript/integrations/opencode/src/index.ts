/**
 * MLflow Tracing Plugin for OpenCode V2
 *
 * This plugin listens for session execution completion events and creates MLflow traces
 * directly using the @mlflow/core TypeScript SDK.
 *
 * Usage:
 *   1. Install: npm install @mlflow/opencode
 *   2. Add to opencode.json:
 *      { "plugins": [{ "package": "@mlflow/opencode",
 *                      "options": { "trackingUri": "http://localhost:5000", "experimentId": "123" } }] }
 *      or set MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID in the OpenCode server environment.
 *      # For a UC-backed experiment: export MLFLOW_TRACE_LOCATION=catalog.schema.table_prefix
 *   3. Run opencode normally - tracing happens automatically
 */

import { setTimeout } from 'node:timers/promises';
import {
  init,
  startSpan,
  withSpan,
  updateCurrentTrace,
  flushTraces,
  SpanType,
  SpanAttributeKey,
  SpanStatusCode,
  TokenUsageKey,
  NO_OP_SPAN_TRACE_ID,
} from '@mlflow/core';
import {
  isAssistantMessage,
  isUserMessage,
  type AssistantMessage,
  type Location,
  type ModelCost,
  type PluginDefinition,
  type SessionMessage,
  type TextPart,
  type TokenUsage,
  type ToolPart,
  type ToolState,
} from './opencode.js';
import { MLflowTracingPlugin as warnOpenCodeV1 } from './v1.js';

// Constants
const NANOSECONDS_PER_MS = 1e6;
const MAX_PREVIEW_LENGTH = 1000;
const MAX_TRACKED_SESSIONS = 50;
const RESUBSCRIBE_DELAY_MS = 1000;

// OpenCode V2 no longer emits session.idle. Each run ends with one of these events.
const EVENT_SUCCEEDED = 'session.execution.succeeded';
const EVENT_FAILED = 'session.execution.failed';
const EVENT_INTERRUPTED = 'session.execution.interrupted';
const COMPLETION_EVENTS = [EVENT_SUCCEEDED, EVENT_FAILED, EVENT_INTERRUPTED];

// Well-known trace metadata keys. We pass these through updateCurrentTrace's
// generic `metadata` option rather than its `sessionId`/`user` convenience
// options, since those options (and the TraceMetadataKey.TRACE_SESSION/TRACE_USER
// enum members) are only present in newer @mlflow/core builds. The metadata keys
// themselves are stable, so this works across the published version range.
const TRACE_SESSION_METADATA_KEY = 'mlflow.trace.session';
const TRACE_USER_METADATA_KEY = 'mlflow.trace.user';
// Match Python's SpanAttributeKey.LLM_COST and TraceMetadataKey.COST.
const LLM_COST_ATTRIBUTE_KEY = 'mlflow.llm.cost';
const TRACE_COST_METADATA_KEY = 'mlflow.trace.cost';

// Avoid routine console output so the plugin does not interrupt the TUI.
let debug = process.env.MLFLOW_OPENCODE_DEBUG === 'true';

// Keep local: integrations are independent packages and must support older @mlflow/core floors.
function getCurrentUser(): string {
  return process.env.USER || process.env.USERNAME || '';
}

interface TracingOptions {
  trackingUri?: string;
  experimentId?: string;
}

interface LlmCost {
  input_cost?: number;
  output_cost?: number;
  total_cost: number;
}

type Outcome =
  | { status: 'completed' }
  | { status: 'interrupted' }
  | { status: 'failed'; error: string };

// SDK initialization state
let initialized = false;
let lastInitializationWarning: string | null = null;
let inactiveTracerWarned = false;

function stringOption(value: unknown): string | undefined {
  if (typeof value === 'string' && value) {
    return value;
  }
  return typeof value === 'number' ? String(value) : undefined;
}

/**
 * Initialize the MLflow tracing SDK if not already initialized.
 * Plugin options take precedence over the MLFLOW_TRACKING_URI and
 * MLFLOW_EXPERIMENT_ID environment variables. MLFLOW_TRACE_LOCATION is
 * optional and resolved by the core SDK. If init() fails, a later event
 * retries it; repeated warnings are suppressed.
 */
function ensureInitialized(options: TracingOptions): boolean {
  if (initialized) {
    return true;
  }

  const trackingUri = options.trackingUri ?? process.env.MLFLOW_TRACKING_URI;
  const experimentId = options.experimentId ?? process.env.MLFLOW_EXPERIMENT_ID;

  if (!trackingUri) {
    if (debug) {
      console.error('[mlflow] MLFLOW_TRACKING_URI not set, skipping initialization');
    }
    return false;
  }

  if (!experimentId) {
    if (debug) {
      console.error('[mlflow] MLFLOW_EXPERIMENT_ID not set, skipping initialization');
    }
    return false;
  }

  try {
    init({ trackingUri, experimentId });
    initialized = true;
    lastInitializationWarning = null;
    if (debug) {
      console.error('[mlflow] SDK initialized successfully');
    }
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message !== lastInitializationWarning) {
      console.error(`[mlflow] OpenCode tracing is disabled: ${message}`);
      lastInitializationWarning = message;
    }
    return false;
  }
}

/**
 * Convert timestamp in milliseconds to nanoseconds
 */
function timestampToNs(timestamp: number | undefined): number | undefined {
  return timestamp != null ? Math.floor(timestamp * NANOSECONDS_PER_MS) : undefined;
}

function joinText(parts: AssistantMessage['content'], type: TextPart['type']): string {
  return parts
    .filter((p): p is TextPart => p.type === type && Boolean(p.text))
    .map((p) => p.text)
    .join(type === 'reasoning' ? '\n\n' : '\n');
}

function toolParts(message: AssistantMessage): ToolPart[] {
  return message.content.filter((p): p is ToolPart => p.type === 'tool');
}

function toolInput(state: ToolState): Record<string, unknown> {
  if (typeof state.input === 'string') {
    return { arguments: state.input };
  }
  return { ...(state.input ?? {}) };
}

function toolOutput(state: ToolState): string {
  return (state.content ?? [])
    .map((item) => (item.type === 'text' ? item.text : `[file: ${item.uri}]`))
    .join('\n');
}

/**
 * Extract the assistant response from messages (find last assistant message text).
 * Concatenates all text parts to handle multi-part assistant responses.
 */
function extractAssistantResponse(messages: SessionMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (isAssistantMessage(msg)) {
      const text = msg.content
        .filter((p): p is TextPart => p.type === 'text' && Boolean(p.text?.trim()))
        .map((p) => p.text)
        .join('\n');
      if (text) {
        return text;
      }
    }
  }
  return '';
}

/**
 * Find the index of the last user message
 */
function findLastUserMessageIndex(messages: SessionMessage[]): number | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (isUserMessage(messages[i])) {
      return i;
    }
  }
  return null;
}

/**
 * Reconstruct conversation messages for LLM input history
 */
function reconstructConversationMessages(
  messages: SessionMessage[],
  endIdx: number,
): Array<{ role: string; content: string; reasoning?: string; tool_call_id?: string }> {
  const result: Array<{
    role: string;
    content: string;
    reasoning?: string;
    tool_call_id?: string;
  }> = [];

  for (let i = 0; i < endIdx; i++) {
    const msg = messages[i];

    if (isUserMessage(msg)) {
      if (msg.text) {
        result.push({ role: 'user', content: msg.text });
      }
    } else if (isAssistantMessage(msg)) {
      const content = joinText(msg.content, 'text');
      const reasoning = joinText(msg.content, 'reasoning');
      if (content || reasoning) {
        result.push({ role: 'assistant', content, ...(reasoning ? { reasoning } : {}) });
      }

      // Add tool results as tool messages
      for (const part of toolParts(msg)) {
        if (part.state?.status === 'completed') {
          result.push({ role: 'tool', tool_call_id: part.id, content: toolOutput(part.state) });
        }
      }
    }
  }

  return result;
}

/**
 * Build token usage object for LLM spans
 */
function buildTokenUsage(tokens: TokenUsage | undefined): Record<string, number> | null {
  if (!tokens) {
    return null;
  }

  // OpenCode reports input without cache tokens and output without reasoning
  // tokens. MLflow counts both in input_tokens and output_tokens.
  const cacheRead = tokens.cache?.read || 0;
  const cacheWrite = tokens.cache?.write || 0;
  const inputTokens = (tokens.input || 0) + cacheRead + cacheWrite;
  const outputTokens = (tokens.output || 0) + (tokens.reasoning || 0);

  const usage: Record<string, number> = {
    [TokenUsageKey.INPUT_TOKENS]: inputTokens,
    [TokenUsageKey.OUTPUT_TOKENS]: outputTokens,
    [TokenUsageKey.TOTAL_TOKENS]: inputTokens + outputTokens,
  };
  if (cacheRead) {
    usage[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = cacheRead;
  }
  if (cacheWrite) {
    usage[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] = cacheWrite;
  }

  return usage;
}

/**
 * Build the cost object for LLM spans from the cost that OpenCode reports.
 *
 * OpenCode reports only the total cost of a model call. The MLflow UI shows
 * cost only with an input/output breakdown, so split the total by the model's
 * base token prices. Without prices, record only the total.
 */
function buildCost(msg: AssistantMessage, price: ModelCost | undefined): LlmCost | null {
  if (typeof msg.cost !== 'number') {
    return null;
  }

  const tokens = msg.tokens ?? {};
  const inputPart = price
    ? (tokens.input || 0) * price.input +
      (tokens.cache?.read || 0) * (price.cache?.read ?? price.input) +
      (tokens.cache?.write || 0) * (price.cache?.write ?? price.input)
    : 0;
  const outputPart = price ? ((tokens.output || 0) + (tokens.reasoning || 0)) * price.output : 0;
  if (inputPart + outputPart <= 0) {
    return msg.cost === 0
      ? { input_cost: 0, output_cost: 0, total_cost: 0 }
      : { total_cost: msg.cost };
  }

  const inputCost = (msg.cost * inputPart) / (inputPart + outputPart);
  return { input_cost: inputCost, output_cost: msg.cost - inputCost, total_cost: msg.cost };
}

function sumCosts(costs: LlmCost[]): LlmCost | null {
  if (costs.length === 0) {
    return null;
  }
  const total: LlmCost = { total_cost: costs.reduce((sum, cost) => sum + cost.total_cost, 0) };
  if (costs.every((cost) => cost.input_cost != null && cost.output_cost != null)) {
    total.input_cost = costs.reduce((sum, cost) => sum + (cost.input_cost ?? 0), 0);
    total.output_cost = costs.reduce((sum, cost) => sum + (cost.output_cost ?? 0), 0);
  }
  return total;
}

/**
 * Create LLM and tool spans as children of the parent span.
 * Returns the total cost of the model calls, if OpenCode reported one.
 */
function createLlmAndToolSpans(
  parentSpan: ReturnType<typeof startSpan>,
  messages: SessionMessage[],
  startIdx: number,
  prices: ReadonlyMap<string, ModelCost>,
): LlmCost | null {
  const costs: LlmCost[] = [];

  for (let i = startIdx; i < messages.length; i++) {
    const msg = messages[i];
    if (!isAssistantMessage(msg)) {
      continue;
    }

    const modelId = msg.model?.id || 'unknown';
    const providerId = msg.model?.providerID || 'unknown';
    const tools = toolParts(msg);
    const textContent = joinText(msg.content, 'text');
    const reasoningText = joinText(msg.content, 'reasoning');

    // Create LLM span for all assistant messages with content.
    // Tool-call-only responses (no text) are still LLM calls and must be traced;
    // omitting them causes missing spans when agents like prometheus issue many
    // back-to-back tool calls without intermediate text.
    if (msg.content.length > 0) {
      const llmSpan = startSpan({
        name: 'llm_call',
        parent: parentSpan,
        spanType: SpanType.LLM,
        startTimeNs: timestampToNs(msg.time?.created),
        inputs: {
          model: `${providerId}/${modelId}`,
          messages: reconstructConversationMessages(messages, i),
        },
        attributes: {
          model: modelId,
          provider: providerId,
        },
      });

      const tokenUsage = buildTokenUsage(msg.tokens);
      if (tokenUsage) {
        llmSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, tokenUsage);
      }
      const cost = buildCost(msg, prices.get(`${providerId}/${modelId}`));
      if (cost) {
        llmSpan.setAttribute(LLM_COST_ATTRIBUTE_KEY, cost);
        costs.push(cost);
      }

      // Emit the OpenAI chat-completion message shape (content: string | null,
      // tool_calls: [{id, type, function: {name, arguments}}]) so MLflow's Chat view
      // renders tool-call spans and traces stay consistent with the other integrations
      // (see integrations/codex and integrations/qwen-code).
      const outputMessage: {
        role: string;
        content: string | null;
        reasoning?: string;
        tool_calls?: Array<{
          id: string;
          type: 'function';
          function: { name: string; arguments: string };
        }>;
      } = { role: 'assistant', content: textContent || null };
      if (reasoningText) {
        outputMessage.reasoning = reasoningText;
      }
      if (tools.length > 0) {
        outputMessage.tool_calls = tools.map((p) => ({
          id: p.id || '',
          type: 'function' as const,
          function: {
            name: p.name || 'unknown',
            arguments: JSON.stringify(toolInput(p.state ?? {})),
          },
        }));
      }

      llmSpan.setOutputs({ choices: [{ message: outputMessage }] });
      llmSpan.end({ endTimeNs: timestampToNs(msg.time?.completed) });
    }

    // Create tool spans
    for (const toolPart of tools) {
      const state = toolPart.state ?? {};
      const toolName = toolPart.name || 'unknown';

      const toolSpan = startSpan({
        name: `tool_${toolName}`,
        parent: parentSpan,
        spanType: SpanType.TOOL,
        startTimeNs: timestampToNs(toolPart.time?.ran ?? toolPart.time?.created),
        inputs: toolInput(state),
        attributes: {
          tool_name: toolName,
          tool_id: toolPart.id || '',
          status: state.status || 'unknown',
        },
      });

      // Set output based on status
      if (state.status === 'completed') {
        const title = state.metadata?.title;
        // OpenCode V2 sets a title only for some tools.
        toolSpan.setOutputs({
          result: toolOutput(state),
          ...(typeof title === 'string' && title ? { title } : {}),
        });
      } else if (state.status === 'error') {
        toolSpan.setOutputs({
          error: state.error?.message || 'Unknown error',
        });
      }

      toolSpan.end({
        endTimeNs: timestampToNs(toolPart.time?.completed),
        ...(state.status === 'error' ? { status: SpanStatusCode.ERROR } : {}),
      });
    }
  }

  return sumCosts(costs);
}

/**
 * Process a session and create MLflow traces
 */
async function processSession(
  sessionId: string,
  messages: SessionMessage[],
  outcome: Outcome,
  prices: ReadonlyMap<string, ModelCost> = new Map(),
): Promise<void> {
  if (!messages || messages.length === 0) {
    if (debug) {
      console.error('[mlflow] Empty messages list, skipping');
    }
    return;
  }

  const lastUserIdx = findLastUserMessageIndex(messages);
  if (lastUserIdx == null) {
    if (debug) {
      console.error('[mlflow] No user message found in session');
    }
    return;
  }

  const userPrompt = (messages[lastUserIdx] as { text?: string }).text ?? '';
  if (!userPrompt) {
    if (debug) {
      console.error('[mlflow] No user prompt text found');
    }
    return;
  }

  if (debug) {
    console.error(`[mlflow] Creating trace for session: ${sessionId}`);
  }

  // Get timing from the FIRST message in this batch
  const firstMsg = messages[0] as { time?: { created?: number } };
  const createdNs = timestampToNs(firstMsg.time?.created);

  // Get end time from the LAST message
  const lastMsg = messages[messages.length - 1] as {
    time?: { created?: number; completed?: number };
  };
  const updatedNs = timestampToNs(lastMsg.time?.completed ?? lastMsg.time?.created);

  // Get final response for preview
  const finalResponse = extractAssistantResponse(messages);

  // Create the parent span via withSpan so it becomes the OTel context-active
  // span. This lets us attach trace-level metadata through the public
  // `updateCurrentTrace` API (which targets the active trace) instead of
  // reaching into `InMemoryTraceManager` internals.
  const traceId = await withSpan(
    (parentSpan) => {
      // Create child spans for LLM calls and tools
      const totalCost = createLlmAndToolSpans(parentSpan, messages, lastUserIdx + 1, prices);

      // Attach session/user, cost, and request/response previews to the trace.
      updateCurrentTrace({
        metadata: {
          [TRACE_SESSION_METADATA_KEY]: sessionId,
          [TRACE_USER_METADATA_KEY]: getCurrentUser(),
          ...(totalCost != null ? { [TRACE_COST_METADATA_KEY]: JSON.stringify(totalCost) } : {}),
        },
        requestPreview: userPrompt.slice(0, MAX_PREVIEW_LENGTH),
        ...(finalResponse ? { responsePreview: finalResponse.slice(0, MAX_PREVIEW_LENGTH) } : {}),
      });

      // End the parent span with the reconstructed end time. withSpan issues a
      // trailing end() at the current time, but OpenTelemetry ignores end() on
      // an already-ended span, so this explicit end time is what is recorded.
      if (outcome.status === 'failed') {
        parentSpan.setStatus(SpanStatusCode.ERROR, outcome.error);
      }
      parentSpan.end({
        outputs: {
          response: finalResponse || 'Conversation completed',
          status: outcome.status,
          ...(outcome.status === 'failed' ? { error: outcome.error } : {}),
        },
        endTimeNs: updatedNs,
      });

      return parentSpan.traceId;
    },
    {
      name: 'opencode_conversation',
      inputs: { prompt: userPrompt },
      startTimeNs: createdNs,
      spanType: SpanType.AGENT,
    },
  );

  // Flush traces to MLflow
  await flushTraces();

  // OpenTelemetry registers its tracer provider once per process. When OpenCode
  // reloads the plugin module, the new copy of the SDK cannot register again
  // and every span becomes a no-op.
  if (traceId === NO_OP_SPAN_TRACE_ID) {
    if (!inactiveTracerWarned) {
      console.error(
        '[mlflow] OpenCode tracing stopped because the MLflow tracer is inactive, ' +
          'usually after OpenCode reloaded the plugin. Restart OpenCode ' +
          '(`opencode service restart`) to resume tracing.',
      );
      inactiveTracerWarned = true;
    }
    return;
  }

  if (debug) {
    console.error(`[mlflow] Created trace: ${traceId}`);
  }
}

function outcomeOf(event: { type: string; data?: { error?: { message?: string } } }): Outcome {
  if (event.type === EVENT_FAILED) {
    return { status: 'failed', error: event.data?.error?.message || 'Unknown error' };
  }
  return event.type === EVENT_INTERRUPTED ? { status: 'interrupted' } : { status: 'completed' };
}

/**
 * MLflow tracing plugin for OpenCode V2.
 * Automatically traces conversations to MLflow when session execution ends.
 */
export const MLflowTracingPlugin: PluginDefinition & { server: typeof warnOpenCodeV1 } = {
  id: 'mlflow.tracing',
  // Newer OpenCode V1 releases also load the `./server` export and call
  // `server()`. OpenCode V2 only calls `setup()`.
  server: warnOpenCodeV1,
  setup(ctx) {
    const options = ctx.options ?? {};
    const tracingOptions: TracingOptions = {
      trackingUri: stringOption(options.trackingUri),
      experimentId: stringOption(options.experimentId),
    };
    debug ||= options.debug === true;

    const controller = new AbortController();
    const { signal } = controller;
    // Last traced message ID per session. IDs survive context compaction,
    // unlike message counts.
    const processedMessages = new Map<string, string>();
    const pending = new Map<string, Promise<void>>();
    // Base token prices per "provider/model", and models without known prices.
    const prices = new Map<string, ModelCost>();
    const unpriced = new Set<string>();

    const loadPrices = async (messages: SessionMessage[]): Promise<void> => {
      const missing = new Set(
        messages
          .filter(isAssistantMessage)
          .filter((msg) => (msg.cost ?? 0) > 0)
          .map((msg) => `${msg.model?.providerID}/${msg.model?.id}`)
          .filter((key) => !prices.has(key) && !unpriced.has(key)),
      );
      if (missing.size === 0 || !ctx.model) {
        return;
      }
      try {
        const { data } = await ctx.model.list(
          { location: { directory: ctx.location.directory } },
          { signal },
        );
        for (const model of data) {
          const price = model.cost?.find((cost) => !cost.tier) ?? model.cost?.[0];
          if (price) {
            prices.set(`${model.providerID}/${model.id}`, price);
          }
        }
      } catch {
        // Without prices the trace still records each call's total cost.
      }
      for (const key of missing) {
        if (!prices.has(key)) {
          unpriced.add(key);
        }
      }
    };

    const processCompletion = async (
      sessionID: string,
      location: Location | undefined,
      outcome: Outcome,
    ): Promise<void> => {
      if (signal.aborted) {
        return;
      }

      try {
        // Execution events can omit location. Resolve ownership from the
        // session instead of exporting other projects' conversations through
        // this plugin instance.
        const sessionLocation =
          location ?? (await ctx.session.get({ sessionID }, { signal })).location;
        if (sessionLocation.directory !== ctx.location.directory) {
          return;
        }
        if (location && location.workspaceID !== ctx.location.workspaceID) {
          return;
        }
        if (signal.aborted || !ensureInitialized(tracingOptions)) {
          return;
        }
        const messages = await ctx.session.context({ sessionID }, { signal });
        if (signal.aborted || messages.length === 0) {
          return;
        }
        const lastID = messages[messages.length - 1].id;
        const previousID = processedMessages.get(sessionID);
        if (previousID) {
          // Refresh the LRU position so active sessions are not evicted.
          processedMessages.delete(sessionID);
          processedMessages.set(sessionID, previousID);
        }
        if (lastID === previousID) {
          return;
        }

        // Only commit after processing completes so thrown errors allow a later retry.
        const start = messages.findIndex((message) => message.id === previousID) + 1;
        const turn = messages.slice(start);
        await loadPrices(turn);
        await processSession(sessionID, turn, outcome, prices);
        processedMessages.set(sessionID, lastID);
        while (processedMessages.size > MAX_TRACKED_SESSIONS) {
          const oldest = processedMessages.keys().next().value;
          if (oldest !== undefined) {
            processedMessages.delete(oldest);
          }
        }
      } catch {
        if (debug && !signal.aborted) {
          console.error(
            '[mlflow] Could not trace the completed session; a later completion event will retry',
          );
        }
      }
    };

    const listen = async (): Promise<void> => {
      while (!signal.aborted) {
        try {
          const events = ctx.event.subscribe({ signal, types: COMPLETION_EVENTS });
          for await (const event of events) {
            if (signal.aborted) {
              break;
            }
            const sessionID = event.data?.sessionID;
            if (!COMPLETION_EVENTS.includes(event.type) || !sessionID) {
              continue;
            }
            // The event stream includes other plugin locations too.
            if (
              event.location &&
              (event.location.directory !== ctx.location.directory ||
                event.location.workspaceID !== ctx.location.workspaceID)
            ) {
              continue;
            }

            // Do not hold up the event stream during an MLflow export.
            // Serialize work for each session to avoid concurrent duplicates.
            const outcome = outcomeOf(event);
            const task = (pending.get(sessionID) ?? Promise.resolve()).then(() =>
              processCompletion(sessionID, event.location, outcome),
            );
            pending.set(sessionID, task);
            void task.finally(() => {
              if (pending.get(sessionID) === task) {
                pending.delete(sessionID);
              }
            });
          }
        } catch {
          if (debug && !signal.aborted) {
            console.error('[mlflow] Event subscription ended; reconnecting');
          }
        }
        // V2 subscriptions are live-only and do not reconnect automatically.
        if (!signal.aborted) {
          await setTimeout(RESUBSCRIBE_DELAY_MS, undefined, { signal }).catch(() => {});
        }
      }
    };

    const listener = listen();
    return async () => {
      controller.abort();
      await listener;
      await Promise.all(pending.values());
      processedMessages.clear();
    };
  },
};

export default MLflowTracingPlugin;

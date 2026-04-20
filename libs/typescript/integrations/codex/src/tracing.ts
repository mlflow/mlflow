/**
 * MLflow tracing integration for Codex CLI.
 *
 * Two integration modes:
 *
 * 1. **Notify hook** (recommended): Codex passes turn data as a JSON CLI arg
 *    after each agent turn. Simple, no transcript parsing needed.
 *    Configured via `notify` in config.toml.
 *
 * 2. **Transcript parsing**: Reads the rollout JSONL file for richer data
 *    (tool calls, token usage). Used when transcript_path is available.
 *
 * References:
 * - Notify hook: developers.openai.com/codex/hooks
 * - Protocol types: github.com/openai/codex codex-rs/protocol/src/protocol.rs
 * - Rollout recorder: github.com/openai/codex codex-rs/rollout/src/recorder.rs
 */

import {
  startSpan,
  flushTraces,
  InMemoryTraceManager,
  SpanType,
  SpanAttributeKey,
  TraceMetadataKey,
  TokenUsageKey,
  type LiveSpan,
} from '@mlflow/core';

import type {
  ChatMessage,
  EventMsgPayload,
  NotifyPayload,
  RolloutLine,
  ResponseItemPayload,
} from './types.js';
import {
  parseTimestampToNs,
  extractTextFromContent,
  getTokenUsage,
  getModel,
  buildToolResultMap,
  findTranscriptForThread,
  getLastTurnRecords,
  readTranscript,
} from './transcript.js';

/**
 * Process a Codex notify hook payload and create an MLflow trace.
 *
 * The notify payload has the user prompt and assistant response directly,
 * so we create a simple AGENT → LLM trace. If a transcript file is found,
 * we also parse it for tool calls and token usage.
 */
export async function processNotify(payload: NotifyPayload): Promise<void> {
  // input-messages accumulates all prompts in the session; take only the last one
  const inputMessages = payload['input-messages'] ?? [];
  const userPrompt = inputMessages[inputMessages.length - 1] ?? '';
  const assistantResponse = payload['last-assistant-message'] ?? '';
  const sessionId = payload['thread-id'];

  if (!userPrompt) {
    return;
  }

  // Try to find and parse the transcript for richer data (tool calls, tokens)
  const transcriptPath = findTranscriptForThread(sessionId);
  let turnRecords: RolloutLine[] | null = null;
  let model = 'unknown';

  if (transcriptPath) {
    const records = readTranscript(transcriptPath);
    if (records.length > 0) {
      turnRecords = getLastTurnRecords(records);
      model = getModel(records);
    }
  }

  // Create root AGENT span. Pass the user prompt as a raw string so MLflow
  // can auto-generate the request preview and the session view renders the
  // message cleanly.
  const rootSpan = startSpan({
    name: 'codex_conversation',
    spanType: SpanType.AGENT,
    inputs: userPrompt,
    attributes: { model },
  });

  // If we have transcript data, create detailed child spans
  if (turnRecords && turnRecords.length > 0) {
    createChildSpans(rootSpan, turnRecords, model);

    const tokenUsage = getTokenUsage(turnRecords);
    if (tokenUsage) {
      rootSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, {
        [TokenUsageKey.INPUT_TOKENS]: tokenUsage.input_tokens,
        [TokenUsageKey.OUTPUT_TOKENS]: tokenUsage.output_tokens,
        [TokenUsageKey.TOTAL_TOKENS]: tokenUsage.total_tokens,
      });
    }
  } else {
    // Fallback: create a simple LLM span from the notify data using the same
    // OpenAI chat format the transcript path produces.
    const llmSpan = startSpan({
      name: 'llm_call',
      parent: rootSpan,
      spanType: SpanType.LLM,
      inputs: {
        model,
        messages: [{ role: 'user', content: userPrompt }],
      },
      attributes: { model },
    });
    llmSpan.end({
      outputs: {
        choices: [{ message: { role: 'assistant', content: assistantResponse } }],
      },
    });
  }

  // Attach session/user metadata to the trace. We use InMemoryTraceManager
  // directly because `updateCurrentTrace()` requires an active OTel span
  // context, which hook-based integrations don't have — spans are created
  // via `startSpan()` without OTel context propagation.
  const traceId = rootSpan.traceId;
  if (traceId) {
    const traceManager = InMemoryTraceManager.getInstance();
    const trace = traceManager.getTrace(traceId);
    if (trace) {
      trace.info.traceMetadata = {
        ...trace.info.traceMetadata,
        [TraceMetadataKey.TRACE_SESSION]: sessionId,
        [TraceMetadataKey.TRACE_USER]: process.env.USER ?? '',
      };
    }
  }

  rootSpan.end({ outputs: assistantResponse });

  await flushTraces();
}

/**
 * Reconstruct OpenAI chat-format message history from response_items preceding
 * the current index. Used to populate LLM span inputs so the MLflow Chat view
 * shows the full conversation context that led up to each assistant call.
 *
 * Maps Codex's Responses-API-style records to standard chat messages:
 * - `message` (user/assistant/system) → `{role, content}`
 * - `function_call` → assistant message with `tool_calls: [{id, type, function}]`
 * - `function_call_output` → `{role: 'tool', tool_call_id, content}`
 */
export function reconstructMessages(
  responseItems: RolloutLine[],
  uptoIndex: number,
): ChatMessage[] {
  const messages: ChatMessage[] = [];
  for (let i = 0; i < uptoIndex; i++) {
    const payload = responseItems[i].payload as ResponseItemPayload;

    if (payload.type === 'message') {
      const text = extractTextFromContent(payload.content);
      if (!text.trim()) {
        continue;
      }
      if (payload.role === 'user' || payload.role === 'assistant') {
        messages.push({ role: payload.role, content: text });
      } else if (payload.role === 'developer') {
        // Codex uses "developer" for system-style instructions; render as system
        messages.push({ role: 'system', content: text });
      }
    } else if (payload.type === 'function_call') {
      messages.push({
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: payload.call_id ?? '',
            type: 'function',
            function: {
              name: payload.name ?? 'unknown',
              arguments: payload.arguments ?? '{}',
            },
          },
        ],
      });
    } else if (payload.type === 'function_call_output') {
      messages.push({
        role: 'tool',
        tool_call_id: payload.call_id ?? '',
        content: payload.output ?? '',
      });
    }
  }
  return messages;
}

/**
 * Create LLM and TOOL child spans from transcript turn records.
 *
 * Timing model:
 * - LLM span covers "LLM thinking": from the last boundary (turn start or
 *   the previous `function_call_output`) to the `message/assistant` record.
 * - TOOL span covers the actual tool call: from the `function_call` record
 *   to the matching `function_call_output` record (matched by call_id).
 *
 * Using the record's own timestamp as both start and end — or chaining to
 * the next response_item — would produce spans that represent "time between
 * records" rather than the work each span describes. The record's timestamp
 * marks when the event was logged, which for an assistant message is when
 * generation *finished*, not when it started.
 */
export function createChildSpans(
  parentSpan: LiveSpan,
  turnRecords: RolloutLine[],
  model: string,
): void {
  const toolResults = buildToolResultMap(turnRecords);
  const toolEndTimes = buildToolEndTimes(turnRecords);

  // Initial boundary for the first LLM span: the turn's task_started event,
  // if present. Falls back to null so the LLM span omits startTimeNs.
  let prevBoundaryNs: number | null = findTaskStartedNs(turnRecords);

  const responseItems = turnRecords.filter((record) => record.type === 'response_item');

  for (let i = 0; i < responseItems.length; i++) {
    const record = responseItems[i];
    const payload = record.payload as ResponseItemPayload;
    const timestampNs = parseTimestampToNs(record.timestamp);
    if (timestampNs == null) {
      continue;
    }

    if (payload.type === 'message' && payload.role === 'assistant') {
      const text = extractTextFromContent(payload.content);
      if (text.trim()) {
        const messages = reconstructMessages(responseItems, i);
        const llmSpan = startSpan({
          name: 'llm_call',
          parent: parentSpan,
          spanType: SpanType.LLM,
          startTimeNs: prevBoundaryNs ?? timestampNs,
          inputs: { model, messages },
          attributes: { model },
        });
        llmSpan.end({
          outputs: {
            choices: [{ message: { role: 'assistant', content: text } }],
          },
          endTimeNs: timestampNs,
        });
        prevBoundaryNs = timestampNs;
      }
    } else if (payload.type === 'function_call') {
      const callId = payload.call_id ?? '';
      const funcName = payload.name ?? 'unknown';
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(payload.arguments ?? '{}');
      } catch {
        // keep empty
      }

      const toolSpan = startSpan({
        name: `tool_${funcName}`,
        parent: parentSpan,
        spanType: SpanType.TOOL,
        startTimeNs: timestampNs,
        inputs: args,
        attributes: { tool_name: funcName, tool_id: callId },
      });
      toolSpan.end({
        outputs: { result: toolResults[callId] ?? '' },
        endTimeNs: toolEndTimes[callId] ?? timestampNs,
      });
    } else if (payload.type === 'function_call_output') {
      // Tool result logged; the next LLM span should start from here, since
      // the LLM is waiting on tool output until this point.
      prevBoundaryNs = timestampNs;
    }
  }
}

/**
 * Find the turn's `task_started` event_msg timestamp in nanoseconds.
 * Returns null if the turn doesn't include a task_started event.
 */
function findTaskStartedNs(turnRecords: RolloutLine[]): number | null {
  for (const record of turnRecords) {
    if (record.type === 'event_msg') {
      const payload = record.payload as EventMsgPayload;
      if (payload.type === 'task_started') {
        return parseTimestampToNs(record.timestamp);
      }
    }
  }
  return null;
}

/**
 * Build a lookup from function call_id to its `function_call_output`
 * timestamp (ns). Used to derive accurate TOOL span end times instead of
 * chaining to the next response_item, which may not be the matching output.
 */
function buildToolEndTimes(turnRecords: RolloutLine[]): Record<string, number> {
  const endTimes: Record<string, number> = {};
  for (const record of turnRecords) {
    if (record.type !== 'response_item') {
      continue;
    }
    const payload = record.payload as ResponseItemPayload;
    if (payload.type === 'function_call_output' && payload.call_id) {
      const ts = parseTimestampToNs(record.timestamp);
      if (ts != null) {
        endTimes[payload.call_id] = ts;
      }
    }
  }
  return endTimes;
}

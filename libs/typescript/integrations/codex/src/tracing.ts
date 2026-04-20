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

import type { NotifyPayload, RolloutLine, ResponseItemPayload } from './types.js';
import {
  NANOSECONDS_PER_MS,
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
    // Fallback: create a simple LLM span from the notify data
    const llmSpan = startSpan({
      name: 'llm',
      parent: rootSpan,
      spanType: SpanType.LLM,
      inputs: { prompt: userPrompt },
    });
    llmSpan.end({
      outputs: { content: assistantResponse },
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
 * Create LLM and TOOL child spans from transcript turn records.
 */
function createChildSpans(parentSpan: LiveSpan, turnRecords: RolloutLine[], model: string): void {
  const toolResults = buildToolResultMap(turnRecords);

  const responseItems = turnRecords.filter((record) => record.type === 'response_item');

  for (let i = 0; i < responseItems.length; i++) {
    const record = responseItems[i];
    const payload = record.payload as ResponseItemPayload;
    const timestampNs = parseTimestampToNs(record.timestamp);
    if (timestampNs == null) {
      continue;
    }

    // Derive end time from next record or default
    let endTimeNs: number | null = null;
    for (let j = i + 1; j < responseItems.length; j++) {
      const nextTs = parseTimestampToNs(responseItems[j].timestamp);
      if (nextTs != null && nextTs > timestampNs) {
        endTimeNs = nextTs;
        break;
      }
    }
    if (endTimeNs == null) {
      endTimeNs = timestampNs + 1000 * NANOSECONDS_PER_MS;
    }

    if (payload.type === 'message' && payload.role === 'assistant') {
      const text = extractTextFromContent(payload.content);
      if (text.trim()) {
        const llmSpan = startSpan({
          name: 'llm',
          parent: parentSpan,
          spanType: SpanType.LLM,
          startTimeNs: timestampNs,
          inputs: { model },
          attributes: { model },
        });
        llmSpan.end({ outputs: { content: text }, endTimeNs });
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
        endTimeNs,
      });
    }
  }
}

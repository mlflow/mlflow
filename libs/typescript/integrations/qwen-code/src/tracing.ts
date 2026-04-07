/**
 * MLflow tracing integration for Qwen Code.
 *
 * Parses Qwen Code JSONL transcripts (tree-structured ChatRecords)
 * and creates MLflow traces with AGENT → LLM/TOOL span hierarchies.
 *
 * Qwen Code fires a Stop hook via stdin with {session_id, transcript_path}.
 * Requires hooksConfig.enabled=true in .qwen/settings.json.
 *
 * Reference: github.com/QwenLM/qwen-code
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

import type { ChatRecord } from './types.js';
import {
  readTranscript,
  parseTimestampToNs,
  getMessageText,
  findLastUserRecord,
  buildRecordTree,
  getTokenUsage,
} from './transcript.js';

const NANOSECONDS_PER_MS = 1e6;
const MAX_PREVIEW_LENGTH = 1000;

/**
 * Process a Qwen Code transcript and create an MLflow trace.
 */
export async function processTranscript(
  transcriptPath: string | null,
  sessionId?: string,
): Promise<void> {
  if (!transcriptPath) {
    return;
  }

  const records = readTranscript(transcriptPath);
  if (records.length === 0) {
    return;
  }

  const userRecord = findLastUserRecord(records);
  if (!userRecord) {
    return;
  }

  const userPrompt = getMessageText(userRecord);
  const resolvedSessionId = sessionId ?? userRecord.sessionId ?? `qwen-${Date.now()}`;
  const model = records.find((r) => r.model)?.model ?? 'unknown';

  const { byUuid, children } = buildRecordTree(records);

  const startNs = parseTimestampToNs(userRecord.timestamp);

  // Create root AGENT span
  const rootSpan = startSpan({
    name: 'qwen_code_conversation',
    spanType: SpanType.AGENT,
    startTimeNs: startNs ?? undefined,
    inputs: { prompt: userPrompt },
    attributes: { model },
  });

  // Create child spans by walking the tree from the user record
  const finalResponse = createChildSpans(rootSpan, userRecord.uuid, byUuid, children, model);

  // Set trace previews and metadata
  const traceId = rootSpan.traceId;
  if (traceId) {
    const traceManager = InMemoryTraceManager.getInstance();
    const trace = traceManager.getTrace(traceId);
    if (trace) {
      trace.info.requestPreview = userPrompt.slice(0, MAX_PREVIEW_LENGTH);
      if (finalResponse) {
        trace.info.responsePreview = finalResponse.slice(0, MAX_PREVIEW_LENGTH);
      }
      trace.info.traceMetadata = {
        ...trace.info.traceMetadata,
        [TraceMetadataKey.TRACE_SESSION]: resolvedSessionId,
        [TraceMetadataKey.TRACE_USER]: process.env.USER ?? '',
      };
    }
  }

  // Calculate end time from last record
  const lastRecord = records[records.length - 1];
  const endNs = parseTimestampToNs(lastRecord.timestamp);
  const endTimeNs =
    endNs != null && startNs != null && endNs > startNs ? endNs : Date.now() * NANOSECONDS_PER_MS;

  rootSpan.end({
    outputs: {
      status: 'completed',
      ...(finalResponse ? { response: finalResponse } : {}),
    },
    endTimeNs,
  });

  await flushTraces();
}

/**
 * Recursively create child spans from the ChatRecord tree.
 * Returns the last assistant text response for trace preview.
 */
function createChildSpans(
  parentSpan: LiveSpan,
  recordUuid: string,
  byUuid: Map<string, ChatRecord>,
  childrenMap: Map<string, string[]>,
  model: string,
): string | null {
  let finalResponse: string | null = null;
  const childUuids = childrenMap.get(recordUuid) ?? [];

  for (let i = 0; i < childUuids.length; i++) {
    const child = byUuid.get(childUuids[i]);
    if (!child) {
      continue;
    }

    const timestampNs = parseTimestampToNs(child.timestamp);
    if (timestampNs == null) {
      continue;
    }

    // Derive end time from next sibling or default
    let endTimeNs: number | null = null;
    for (let j = i + 1; j < childUuids.length; j++) {
      const nextChild = byUuid.get(childUuids[j]);
      if (nextChild) {
        const nextTs = parseTimestampToNs(nextChild.timestamp);
        if (nextTs != null && nextTs > timestampNs) {
          endTimeNs = nextTs;
          break;
        }
      }
    }
    if (endTimeNs == null) {
      endTimeNs = timestampNs + 1000 * NANOSECONDS_PER_MS;
    }

    // Tool call result → TOOL span
    if (child.toolCallResult) {
      const toolName = child.toolCallResult.name ?? 'unknown';
      const toolSpan = startSpan({
        name: `tool_${toolName}`,
        parent: parentSpan,
        spanType: SpanType.TOOL,
        startTimeNs: timestampNs,
        inputs: child.toolCallResult.input ?? {},
        attributes: { tool_name: toolName },
      });
      toolSpan.end({
        outputs: { result: child.toolCallResult.output ?? '' },
        endTimeNs,
      });
    }
    // Assistant text → LLM span
    else if (child.type === 'assistant') {
      const text = getMessageText(child);
      if (text.trim()) {
        finalResponse = text;
        const llmSpan = startSpan({
          name: 'llm',
          parent: parentSpan,
          spanType: SpanType.LLM,
          startTimeNs: timestampNs,
          inputs: { model: child.model ?? model },
          attributes: { model: child.model ?? model },
        });

        const tokenUsage = getTokenUsage(child.usageMetadata);
        if (tokenUsage) {
          llmSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, {
            [TokenUsageKey.INPUT_TOKENS]: tokenUsage.input,
            [TokenUsageKey.OUTPUT_TOKENS]: tokenUsage.output,
            [TokenUsageKey.TOTAL_TOKENS]: tokenUsage.total,
          });
        }

        llmSpan.end({ outputs: { content: text }, endTimeNs });
      }
    }

    // Recurse into children
    const childText = createChildSpans(parentSpan, child.uuid, byUuid, childrenMap, model);
    if (childText) {
      finalResponse = childText;
    }
  }

  return finalResponse;
}

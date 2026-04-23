import { existsSync } from 'node:fs';
import { resolve, dirname, basename } from 'node:path';

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
  ContentBlock,
  SubagentGroup,
  TokenUsage,
  ToolResultInfo,
  TranscriptEntry,
} from './types.js';
import {
  extractTextContent,
  findFinalAssistantResponse,
  findLastUserMessageIndex,
  getNextTimestampNs,
  parseTimestampToNs,
  readTranscript,
} from './transcript.js';

// ============================================================================
// Constants
// ============================================================================

const NANOSECONDS_PER_MS = 1e6;
const NANOSECONDS_PER_S = 1e9;
const MAX_PREVIEW_LENGTH = 1000;
const METADATA_KEY_CLAUDE_CODE_VERSION = 'mlflow.claude_code_version';

// ============================================================================
// Content and tool extraction
// ============================================================================

/**
 * Separate text content from tool_use blocks in an assistant response.
 */
function extractContentAndTools(
  content: string | ContentBlock[],
): [string, Array<{ type: 'tool_use'; id: string; name: string; input: Record<string, unknown> }>] {
  let textContent = '';
  const toolUses: Array<{
    type: 'tool_use';
    id: string;
    name: string;
    input: Record<string, unknown>;
  }> = [];

  if (!Array.isArray(content)) {
    return [typeof content === 'string' ? content : '', toolUses];
  }

  for (const part of content) {
    if (typeof part !== 'object' || part == null || !('type' in part)) {
      continue;
    }
    if (part.type === 'text' && 'text' in part) {
      textContent += (part as { type: 'text'; text: string }).text;
    } else if (part.type === 'tool_use') {
      toolUses.push(
        part as { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> },
      );
    }
  }

  return [textContent, toolUses];
}

// ============================================================================
// Tool result finding
// ============================================================================

/**
 * Find tool results following the current assistant response.
 * Returns a mapping from tool_use_id to result info.
 */
function findToolResults(
  transcript: TranscriptEntry[],
  startIdx: number,
): Record<string, ToolResultInfo> {
  const results: Record<string, ToolResultInfo> = {};
  // Claude Code splits a single assistant turn into multiple JSONL entries
  // (one per content block) that share the same message.id. Treat them as
  // one turn so parallel tool_uses in the same turn all find their results.
  const currentMessageId = transcript[startIdx]?.message?.id;

  for (let i = startIdx + 1; i < transcript.length; i++) {
    const entry = transcript[i];
    if (entry.type === 'assistant') {
      if (currentMessageId && entry.message?.id === currentMessageId) {
        continue;
      }
      break;
    }
    if (entry.type !== 'user') {
      continue;
    }

    // Entry-level toolUseResult (used in real Claude Code transcripts)
    const entryToolUseResult =
      entry.toolUseResult && typeof entry.toolUseResult === 'object' ? entry.toolUseResult : {};

    const content = entry.message?.content;
    if (!Array.isArray(content)) {
      continue;
    }

    for (const part of content) {
      if (typeof part !== 'object' || part == null || !('type' in part)) {
        continue;
      }
      if (part.type !== 'tool_result') {
        continue;
      }

      const toolResult = part as {
        type: 'tool_result';
        tool_use_id?: string;
        content?: string;
        is_error?: boolean;
        toolUseResult?: { agentId?: string };
      };

      const toolUseId = toolResult.tool_use_id;
      if (!toolUseId) {
        continue;
      }

      // Check both entry-level and content-level toolUseResult for agentId
      const partToolUseResult = toolResult.toolUseResult ?? {};
      const agentId = entryToolUseResult.agentId ?? partToolUseResult.agentId;

      results[toolUseId] = {
        content: toolResult.content ?? '',
        isError: toolResult.is_error ?? false,
        agentId,
      };
    }
  }

  return results;
}

// ============================================================================
// Input message reconstruction
// ============================================================================

/**
 * Get all messages between the previous text-bearing assistant response
 * and the current one, for use as LLM span inputs.
 */
function getInputMessages(
  transcript: TranscriptEntry[],
  currentIdx: number,
): Array<{ role: string; content: unknown }> {
  const messages: Array<{ role: string; content: unknown }> = [];

  for (let i = currentIdx - 1; i >= 0; i--) {
    const entry = transcript[i];
    const msg = entry.message;

    // Stop at a previous assistant entry that has text content (previous LLM span)
    if (entry.type === 'assistant' && msg) {
      const content = msg.content;
      let hasText = false;
      if (typeof content === 'string') {
        hasText = content.trim().length > 0;
      } else if (Array.isArray(content)) {
        hasText = content.some(
          (p) => typeof p === 'object' && p != null && 'type' in p && p.type === 'text',
        );
      }
      if (hasText) {
        break;
      }
    }

    // Include steer messages (queue-operation enqueue) as user messages
    if (entry.type === 'queue-operation' && entry.operation === 'enqueue' && entry.content) {
      messages.push({ role: 'user', content: entry.content });
      continue;
    }

    if (msg?.role && msg?.content) {
      messages.push({ role: msg.role, content: msg.content });
    }
  }

  messages.reverse();
  return messages;
}

// ============================================================================
// Token usage
// ============================================================================

/**
 * Set token usage on a span. Input = input_tokens + cache_creation (cache_read excluded).
 */
function setTokenUsageAttribute(span: LiveSpan, usage: TokenUsage | undefined): void {
  if (!usage) {
    return;
  }

  const inputTokens = (usage.input_tokens ?? 0) + (usage.cache_creation_input_tokens ?? 0);
  const outputTokens = usage.output_tokens ?? 0;

  span.setAttribute(SpanAttributeKey.TOKEN_USAGE, {
    [TokenUsageKey.INPUT_TOKENS]: inputTokens,
    [TokenUsageKey.OUTPUT_TOKENS]: outputTokens,
    [TokenUsageKey.TOTAL_TOKENS]: inputTokens + outputTokens,
  });
}

// ============================================================================
// Sub-agent handling
// ============================================================================

/**
 * Group progress entries by parentToolUseID.
 */
function collectSubagentGroups(
  transcript: TranscriptEntry[],
  startIdx: number,
): Record<string, SubagentGroup> {
  const groups: Record<string, SubagentGroup> = {};

  for (let i = startIdx; i < transcript.length; i++) {
    const entry = transcript[i];
    if (entry.type !== 'progress') {
      continue;
    }

    const data = entry.data;
    if (!data?.message || typeof data.message !== 'object') {
      continue;
    }

    const parentToolId = entry.parentToolUseID;
    if (!parentToolId) {
      continue;
    }

    if (!groups[parentToolId]) {
      groups[parentToolId] = {
        prompt: data.prompt ?? '',
        messages: [],
        timestamp: entry.timestamp,
      };
    }

    groups[parentToolId].messages.push(data.message);
  }

  return groups;
}

/**
 * Derive the sub-agent transcript file path from the main transcript path.
 */
function getSubagentTranscriptPath(
  transcriptPath: string | undefined,
  agentId: string | undefined,
): string | null {
  if (!transcriptPath || !agentId) {
    return null;
  }

  // Session dir = main transcript path without .jsonl extension
  const dir = dirname(transcriptPath);
  const base = basename(transcriptPath, '.jsonl');
  const subagentPath = resolve(dir, base, 'subagents', `agent-${agentId}.jsonl`);

  if (existsSync(subagentPath)) {
    return subagentPath;
  }
  return null;
}

/**
 * Create an AGENT wrapper span for a sub-agent's execution.
 */
function createAgentWrapperSpan(
  parentSpan: LiveSpan,
  toolInput: Record<string, unknown>,
  startNs: number,
): LiveSpan {
  const subagentType = (toolInput.subagent_type as string) ?? '';
  const description = (toolInput.description as string) ?? '';
  const prompt = (toolInput.prompt as string) ?? '';
  const agentName = subagentType ? `subagent_${subagentType}` : 'subagent';

  return startSpan({
    name: agentName,
    parent: parentSpan,
    spanType: SpanType.AGENT,
    startTimeNs: startNs,
    inputs: { prompt, description },
    attributes: { subagent_type: subagentType },
  });
}

/**
 * Create LLM and tool spans for a sub-agent's inner messages (progress-based).
 */
function createSubagentSpans(
  parentSpan: LiveSpan,
  group: SubagentGroup,
  startNs: number,
  totalDurationNs: number,
  toolInput: Record<string, unknown>,
): void {
  const innerMessages = group.messages;
  if (!innerMessages.length) {
    return;
  }

  const agentSpan = createAgentWrapperSpan(parentSpan, toolInput, startNs);

  // Find first assistant message index
  let firstAssistantIdx = 0;
  for (let idx = 0; idx < innerMessages.length; idx++) {
    if (innerMessages[idx].type === 'assistant') {
      firstAssistantIdx = idx;
      break;
    }
  }

  createLlmAndToolSpans(agentSpan, innerMessages, firstAssistantIdx);
  agentSpan.end({ endTimeNs: startNs + totalDurationNs });
}

/**
 * Create LLM and tool spans from a sub-agent's separate transcript file.
 */
function createSubagentSpansFromFile(
  parentSpan: LiveSpan,
  subagentTranscriptPath: string,
  startNs: number,
  totalDurationNs: number,
  toolInput: Record<string, unknown>,
): void {
  try {
    const subagentTranscript = readTranscript(subagentTranscriptPath);
    if (!subagentTranscript.length) {
      return;
    }

    const agentSpan = createAgentWrapperSpan(parentSpan, toolInput, startNs);

    let firstAssistantIdx = 0;
    for (let idx = 0; idx < subagentTranscript.length; idx++) {
      if (subagentTranscript[idx].type === 'assistant') {
        firstAssistantIdx = idx;
        break;
      }
    }

    createLlmAndToolSpans(agentSpan, subagentTranscript, firstAssistantIdx, subagentTranscriptPath);
    agentSpan.end({ endTimeNs: startNs + totalDurationNs });
  } catch (err) {
    console.error(
      `[mlflow] Failed to process sub-agent transcript ${subagentTranscriptPath}:`,
      err,
    );
  }
}

// ============================================================================
// Core span creation
// ============================================================================

/**
 * Create LLM and tool spans for assistant responses with proper timing.
 */
function createLlmAndToolSpans(
  parentSpan: LiveSpan,
  transcript: TranscriptEntry[],
  startIdx: number,
  transcriptPath?: string,
): void {
  const subagentGroups = collectSubagentGroups(transcript, startIdx);

  for (let i = startIdx; i < transcript.length; i++) {
    const entry = transcript[i];
    if (entry.type !== 'assistant') {
      continue;
    }

    const timestampNs = parseTimestampToNs(entry.timestamp);
    if (!timestampNs) {
      continue;
    }

    const nextTimestampNs = getNextTimestampNs(transcript, i);
    const durationNs = nextTimestampNs
      ? nextTimestampNs - timestampNs
      : Math.floor(1000 * NANOSECONDS_PER_MS); // 1 second default

    const msg = entry.message;
    if (!msg) {
      continue;
    }
    const content = msg.content ?? [];
    const usage = msg.usage;

    const [textContent, toolUses] = extractContentAndTools(content);

    // Create LLM span if there's text content (no tools)
    if (textContent.trim() && !toolUses.length) {
      const messages = getInputMessages(transcript, i);
      const model = msg.model ?? 'unknown';

      const llmSpan = startSpan({
        name: 'llm',
        parent: parentSpan,
        spanType: SpanType.LLM,
        startTimeNs: timestampNs,
        inputs: { model, messages },
        attributes: {
          model,
          'mlflow.llm.model': model,
          [SpanAttributeKey.MESSAGE_FORMAT]: 'anthropic',
        },
      });

      setTokenUsageAttribute(llmSpan, usage);

      llmSpan.setOutputs({
        type: 'message',
        role: 'assistant',
        content,
      });
      llmSpan.end({ endTimeNs: timestampNs + durationNs });
    }

    // Create tool spans with proportional timing
    if (toolUses.length) {
      const toolResults = findToolResults(transcript, i);
      const toolDurationNs = Math.floor(durationNs / toolUses.length);

      for (let idx = 0; idx < toolUses.length; idx++) {
        const toolUse = toolUses[idx];
        const toolStartNs = timestampNs + idx * toolDurationNs;
        const toolUseId = toolUse.id ?? '';
        const toolResultInfo = toolResults[toolUseId];
        const toolResult = toolResultInfo?.content ?? 'No result found';
        const toolName = toolUse.name ?? 'unknown';

        const toolSpan = startSpan({
          name: `tool_${toolName}`,
          parent: parentSpan,
          spanType: SpanType.TOOL,
          startTimeNs: toolStartNs,
          inputs: toolUse.input ?? {},
          attributes: {
            tool_name: toolName,
            tool_id: toolUseId,
          },
        });

        // If this is a Task tool, try to read sub-agent transcript
        const agentId = toolResultInfo?.agentId;
        const subagentPath = getSubagentTranscriptPath(transcriptPath, agentId);
        const toolInput = toolUse.input ?? {};

        if (subagentPath) {
          createSubagentSpansFromFile(
            toolSpan,
            subagentPath,
            toolStartNs,
            toolDurationNs,
            toolInput,
          );
        } else if (subagentGroups[toolUseId]) {
          createSubagentSpans(
            toolSpan,
            subagentGroups[toolUseId],
            toolStartNs,
            toolDurationNs,
            toolInput,
          );
        }

        toolSpan.setOutputs({ result: toolResult });

        if (toolResultInfo?.isError) {
          const errorMessage = toolResult || 'Tool execution failed';
          toolSpan.recordException(new Error(errorMessage));
        }

        toolSpan.end({ endTimeNs: toolStartNs + toolDurationNs });
      }
    }
  }
}

// ============================================================================
// Main entry point
// ============================================================================

/**
 * Process a Claude conversation transcript and create an MLflow trace with spans.
 */
export async function processTranscript(transcriptPath: string, sessionId?: string): Promise<void> {
  try {
    const transcript = readTranscript(transcriptPath);
    if (!transcript.length) {
      console.error('[mlflow] Empty transcript, skipping');
      return;
    }

    const lastUserIdx = findLastUserMessageIndex(transcript);
    if (lastUserIdx == null) {
      console.error('[mlflow] No user message found in transcript');
      return;
    }

    const lastUserEntry = transcript[lastUserIdx];
    const lastUserPrompt = lastUserEntry.message?.content ?? '';
    const userPromptText = extractTextContent(lastUserPrompt);

    if (!sessionId) {
      sessionId = `claude-${new Date().toISOString().replace(/[:.]/g, '').slice(0, 15)}`;
    }

    const convStartNs = parseTimestampToNs(lastUserEntry.timestamp);

    const parentSpan = startSpan({
      name: 'claude_code_conversation',
      inputs: { prompt: userPromptText },
      startTimeNs: convStartNs ?? undefined,
      spanType: SpanType.AGENT,
    });

    // Create spans for all assistant responses and tool uses
    createLlmAndToolSpans(parentSpan, transcript, lastUserIdx + 1, transcriptPath);

    // Find final response for preview
    const finalResponse = findFinalAssistantResponse(transcript, lastUserIdx + 1);

    // Set trace previews and metadata
    try {
      const traceManager = InMemoryTraceManager.getInstance();
      const trace = traceManager.getTrace(parentSpan.traceId);
      if (trace) {
        if (userPromptText) {
          trace.info.requestPreview = userPromptText.slice(0, MAX_PREVIEW_LENGTH);
        }
        if (finalResponse) {
          trace.info.responsePreview = finalResponse.slice(0, MAX_PREVIEW_LENGTH);
        }

        const metadata: Record<string, string> = {
          ...trace.info.traceMetadata,
          [TraceMetadataKey.TRACE_SESSION]: sessionId,
          [TraceMetadataKey.TRACE_USER]: process.env.USER ?? '',
          'mlflow.trace.working_directory': process.cwd(),
        };

        // Capture permission mode
        const permissionMode = lastUserEntry.permissionMode;
        if (permissionMode) {
          metadata['mlflow.trace.permission_mode'] = permissionMode;
        }

        // Extract Claude Code version from transcript entries
        const claudeCodeVersion = transcript.reduce<string | undefined>(
          (found, entry) => found ?? entry.version,
          undefined,
        );
        if (claudeCodeVersion) {
          metadata[METADATA_KEY_CLAUDE_CODE_VERSION] = claudeCodeVersion;
        }

        trace.info.traceMetadata = metadata;
      }
    } catch (err) {
      console.error('[mlflow] Failed to update trace metadata:', err);
    }

    // Calculate end time
    const lastEntry = transcript[transcript.length - 1];
    let convEndNs = parseTimestampToNs(lastEntry.timestamp);
    if (!convEndNs || (convStartNs && convEndNs <= convStartNs)) {
      convEndNs = (convStartNs ?? 0) + Math.floor(10 * NANOSECONDS_PER_S);
    }

    const outputs: Record<string, string> = { status: 'completed' };
    if (finalResponse) {
      outputs.response = finalResponse;
    }
    parentSpan.setOutputs(outputs);
    parentSpan.end({ endTimeNs: convEndNs });

    await flushTraces();
  } catch (err) {
    console.error('[mlflow] Error processing transcript:', err);
  }
}

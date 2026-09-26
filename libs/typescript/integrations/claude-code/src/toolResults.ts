/**
 * Tool-result collection from Claude Code transcripts: the one place that
 * decides how a user entry's tool_result blocks map to ToolResultInfo.
 */

import type { ToolResultInfo, TranscriptEntry } from './types.js';

/**
 * Collect the tool results carried by a single user entry, keyed by tool_use_id.
 * Non-user entries carry none.
 */
export function toolResultsInEntry(entry: TranscriptEntry): Record<string, ToolResultInfo> {
  const results: Record<string, ToolResultInfo> = {};
  if (entry.type !== 'user') {
    return results;
  }

  // Entry-level toolUseResult (used in real Claude Code transcripts)
  const entryToolUseResult =
    entry.toolUseResult && typeof entry.toolUseResult === 'object' ? entry.toolUseResult : {};

  const content = entry.message?.content;
  if (!Array.isArray(content)) {
    return results;
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

  return results;
}

/**
 * Find tool results following the current assistant response.
 * Returns a mapping from tool_use_id to result info.
 */
export function findToolResults(
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
    Object.assign(results, toolResultsInEntry(entry));
  }

  return results;
}

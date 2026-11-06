/**
 * Tool-result collection from Claude Code transcripts: the one place that
 * decides how a user entry's tool_result blocks map to ToolResultInfo, and
 * which results count as a sub-agent launch.
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

  const content = entry.message?.content;
  if (!Array.isArray(content)) {
    return results;
  }
  const toolResultParts = content.filter(
    (part) => typeof part === 'object' && part != null && part.type === 'tool_result',
  );

  // Entry-level toolUseResult (used in real Claude Code transcripts) describes
  // the entry's single tool result. With several tool_result parts it cannot
  // be attributed to one of them, so only part-level fields apply.
  const entryToolUseResult =
    toolResultParts.length === 1 && entry.toolUseResult && typeof entry.toolUseResult === 'object'
      ? entry.toolUseResult
      : {};

  for (const part of toolResultParts) {
    const toolResult = part as {
      type: 'tool_result';
      tool_use_id?: string;
      content?: string;
      is_error?: boolean;
      toolUseResult?: { agentId?: string; status?: string };
    };

    const toolUseId = toolResult.tool_use_id;
    if (!toolUseId) {
      continue;
    }

    // Check both entry-level and content-level toolUseResult for agent fields
    const partToolUseResult = toolResult.toolUseResult ?? {};

    results[toolUseId] = {
      content: toolResult.content ?? '',
      isError: toolResult.is_error ?? false,
      agentId: entryToolUseResult.agentId ?? partToolUseResult.agentId,
      status: entryToolUseResult.status ?? partToolUseResult.status,
    };
  }

  return results;
}

/**
 * Tools whose result launches a sub-agent: `Agent`, and `Task`, its name in
 * older Claude Code versions. Results of other tools that carry an `agentId`
 * (for example `SendMessage` resuming an agent) are not launches.
 */
export function isAgentLaunchTool(toolName: string | undefined): boolean {
  return toolName === 'Agent' || toolName === 'Task';
}

/**
 * True when the `Agent` tool result is the launch receipt of a background
 * sub-agent (`run_in_background: true`). Decided on
 * `status === 'async_launched'` alone. Claude Code writes that result
 * immediately, while the agent keeps running; the agent is traced on its own
 * by the SubagentStop hook, never inside the parent's Stop trace.
 */
export function isBackgroundLaunch(info: ToolResultInfo): boolean {
  return info.status === 'async_launched';
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

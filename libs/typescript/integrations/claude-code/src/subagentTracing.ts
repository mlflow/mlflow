/**
 * Tracing for background sub-agents (Agent tool with `run_in_background: true`).
 *
 * The parent's Stop hook sees only the launch receipt of a background agent,
 * so it records a marker on the tool span and leaves the agent alone (see
 * `isBackgroundLaunch`). When the agent finishes, Claude Code fires the
 * SubagentStop hook; this module turns the agent's own transcript into a
 * separate trace. Sync agents stay nested inside the parent's Stop trace and
 * are ignored here, as are Claude Code's internal agents (no Agent tool call
 * in the parent). A resumed agent stops more than once and yields one trace
 * per stop, each covering the turn since its latest prompt.
 */

import { existsSync } from 'node:fs';

import { processTranscript } from './tracing.js';
import { isBackgroundLaunch, toolResultsInEntry } from './toolResults.js';
import { readTranscript } from './transcript.js';
import type { SubagentStopHookInput, ToolResultInfo, TranscriptEntry } from './types.js';

const TAG_AGENT_ID = 'mlflow.claude_code.agent_id';
const TAG_AGENT_TYPE = 'mlflow.claude_code.agent_type';
const TAG_PARENT_TOOL_USE_ID = 'mlflow.claude_code.parent_tool_use_id';

interface AgentLaunch {
  toolUseId: string;
  result: ToolResultInfo;
  toolInput?: Record<string, unknown>;
}

/**
 * Find the parent's tool result that launched `agentId`. The first match in
 * transcript order is the launch; later results naming the same agent (for
 * example a SendMessage resume) do not change how the agent was started.
 */
function findAgentLaunch(transcript: TranscriptEntry[], agentId: string): AgentLaunch | undefined {
  for (const entry of transcript) {
    for (const [toolUseId, result] of Object.entries(toolResultsInEntry(entry))) {
      if (result.agentId === agentId) {
        return { toolUseId, result, toolInput: findToolUseInput(transcript, toolUseId) };
      }
    }
  }
  return undefined;
}

function findToolUseInput(
  transcript: TranscriptEntry[],
  toolUseId: string,
): Record<string, unknown> | undefined {
  for (const entry of transcript) {
    const content = entry.type === 'assistant' ? entry.message?.content : undefined;
    if (!Array.isArray(content)) {
      continue;
    }
    for (const part of content) {
      if (part?.type === 'tool_use' && part.id === toolUseId) {
        return part.input;
      }
    }
  }
  return undefined;
}

/**
 * Trace a finished background sub-agent as its own trace (SubagentStop hook).
 * Does nothing for sync, internal, or unknown agents.
 */
export async function processSubagentTranscript(input: SubagentStopHookInput): Promise<void> {
  try {
    const launch = findAgentLaunch(readTranscript(input.transcript_path), input.agent_id);
    if (!launch || !isBackgroundLaunch(launch.result)) {
      // Expected: the parent's Stop trace owns sync agents, and internal
      // agents have no Agent tool call to attach to.
      return;
    }

    if (!input.agent_transcript_path || !existsSync(input.agent_transcript_path)) {
      console.error(
        `[mlflow] Sub-agent transcript not found for agent ${input.agent_id}: ${input.agent_transcript_path}`,
      );
      return;
    }

    const tags: Record<string, string> = {
      [TAG_AGENT_ID]: input.agent_id,
      [TAG_PARENT_TOOL_USE_ID]: launch.toolUseId,
    };
    if (input.agent_type) {
      tags[TAG_AGENT_TYPE] = input.agent_type;
    }

    await processTranscript(input.agent_transcript_path, input.session_id, {
      rootSpanName: input.agent_type ? `subagent_${input.agent_type}` : 'subagent',
      rootInputs: launch.toolInput,
      tags,
    });
  } catch (err) {
    console.error('[mlflow] Error processing sub-agent transcript:', err);
  }
}

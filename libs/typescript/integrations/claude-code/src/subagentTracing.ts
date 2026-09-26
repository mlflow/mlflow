/**
 * Tracing for background sub-agents (Agent tool with `run_in_background: true`).
 *
 * The parent's Stop hook sees only the launch receipt of a background agent,
 * so it records a marker on the tool span and leaves the agent alone (see
 * `isBackgroundLaunch`). Claude Code fires the SubagentStop hook each time
 * the agent ends a turn; this module turns that stop's work in the agent's
 * own transcript into a separate trace. The window is processTranscript's:
 * from the agent's latest plain user entry (launch prompt, SendMessage
 * resume, or task-notification wake-up) to the end, so each stop yields one
 * trace and a resumed agent yields several, all tagged with its agent_id.
 * Sync agents stay nested inside the parent's Stop trace and are ignored
 * here, as are Claude Code's internal agents (no Agent tool call in the
 * parent).
 */

import { existsSync } from 'node:fs';

import { processTranscript } from './tracing.js';
import { isAgentLaunchTool, isBackgroundLaunch, toolResultsInEntry } from './toolResults.js';
import { readTranscript } from './transcript.js';
import type {
  SubagentStopHookInput,
  ToolResultInfo,
  ToolUseBlock,
  TranscriptEntry,
} from './types.js';

const TAG_AGENT_ID = 'mlflow.claude_code.agent_id';
const TAG_AGENT_TYPE = 'mlflow.claude_code.agent_type';
const TAG_PARENT_TOOL_USE_ID = 'mlflow.claude_code.parent_tool_use_id';

export interface AgentLaunch {
  toolUseId: string;
  result: ToolResultInfo;
}

/**
 * Find the parent's `Agent` (legacy: `Task`) tool result that launched
 * `agentId`. Classification follows the launch: the first matching launch in
 * main-transcript order decides sync vs. background, so an agent later
 * resumed in the other mode (SendMessage) keeps its launch classification.
 * Results of other tools never count as a launch, nor does a result whose
 * tool_use cannot be found (its tool name is unknown).
 */
export function findAgentLaunch(
  transcript: TranscriptEntry[],
  agentId: string,
): AgentLaunch | undefined {
  for (const entry of transcript) {
    for (const [toolUseId, result] of Object.entries(toolResultsInEntry(entry))) {
      if (result.agentId !== agentId) {
        continue;
      }
      const toolUse = findToolUse(transcript, toolUseId);
      if (toolUse && isAgentLaunchTool(toolUse.name)) {
        return { toolUseId, result };
      }
    }
  }
  return undefined;
}

function findToolUse(transcript: TranscriptEntry[], toolUseId: string): ToolUseBlock | undefined {
  for (const entry of transcript) {
    const content = entry.type === 'assistant' ? entry.message?.content : undefined;
    if (!Array.isArray(content)) {
      continue;
    }
    for (const part of content) {
      if (part?.type === 'tool_use' && part.id === toolUseId) {
        return part;
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
      // Expected: the parent's Stop trace owns sync agents, internal agents
      // have no Agent tool call, and a sync agent's result may not be written
      // yet when its SubagentStop fires.
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
      tags,
    });
  } catch (err) {
    console.error('[mlflow] Error processing sub-agent transcript:', err);
  }
}

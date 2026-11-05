import { readFileSync } from 'node:fs';

import type { TranscriptEntry } from './types.js';

// ============================================================================
// Constants
// ============================================================================

const NANOSECONDS_PER_MS = 1e6;
const NANOSECONDS_PER_S = 1e9;

// ============================================================================
// JSONL parsing
// ============================================================================

/**
 * Read and parse a Claude Code transcript from a JSONL file.
 *
 * Claude Code may still be writing the last line while a hook reads the file,
 * so a last line that does not parse yet is skipped. Any other unparsable line
 * still throws.
 */
export function readTranscript(path: string): TranscriptEntry[] {
  const content = readFileSync(path, 'utf-8');
  const lines = content.split('\n').filter((line) => line.trim());
  const entries: TranscriptEntry[] = [];
  lines.forEach((line, idx) => {
    try {
      entries.push(JSON.parse(line) as TranscriptEntry);
    } catch (err) {
      if (idx !== lines.length - 1) {
        throw err;
      }
    }
  });
  return entries;
}

// ============================================================================
// Timestamp utilities
// ============================================================================

/**
 * Convert various timestamp formats to nanoseconds since Unix epoch.
 * Handles ISO strings, Unix seconds, milliseconds, and nanoseconds.
 */
export function parseTimestampToNs(timestamp: string | number | undefined | null): number | null {
  if (!timestamp) {
    return null;
  }

  if (typeof timestamp === 'string') {
    try {
      const dt = new Date(timestamp);
      if (isNaN(dt.getTime())) {
        return null;
      }
      return Math.floor(dt.getTime() * NANOSECONDS_PER_MS);
    } catch {
      return null;
    }
  }

  if (typeof timestamp === 'number') {
    // Unix seconds (< 1e10, e.g. 1705312245)
    if (timestamp < 1e10) {
      return Math.floor(timestamp * NANOSECONDS_PER_S);
    }
    // Milliseconds (< 1e13, e.g. 1705312245123)
    if (timestamp < 1e13) {
      return Math.floor(timestamp * NANOSECONDS_PER_MS);
    }
    // Already nanoseconds
    return Math.floor(timestamp);
  }

  return null;
}

// ============================================================================
// Content extraction
// ============================================================================

/**
 * Extract text content from Claude message content (string or content block array).
 */
export function extractTextContent(content: unknown): string {
  if (Array.isArray(content)) {
    const textParts = content
      .filter(
        (part): part is { type: 'text'; text: string } =>
          typeof part === 'object' &&
          part != null &&
          'type' in part &&
          (part as { type: string }).type === 'text',
      )
      .map((part) => part.text);
    return textParts.join('\n');
  }
  if (typeof content === 'string') {
    return content;
  }
  return String(content);
}

// ============================================================================
// Transcript navigation
// ============================================================================

/**
 * Whether the entry at `idx` is an actual user prompt, as opposed to a tool
 * result, a skill injection, local command output or an empty message.
 */
export function isUserPrompt(transcript: TranscriptEntry[], idx: number): boolean {
  const entry = transcript[idx];
  if (entry.type !== 'user' || entry.toolUseResult || entry.isCompactSummary) {
    return false;
  }

  // Skip skill content injections: a user message immediately following
  // a Skill tool result (which has toolUseResult with commandName)
  if (idx > 0) {
    const prevToolResult = transcript[idx - 1].toolUseResult;
    if (
      prevToolResult &&
      typeof prevToolResult === 'object' &&
      'commandName' in prevToolResult &&
      prevToolResult.commandName
    ) {
      return false;
    }
  }

  const msg = entry.message;
  if (!msg) {
    return false;
  }
  const content = msg.content;

  // Skip tool result messages
  if (Array.isArray(content) && content.length > 0) {
    const first = content[0];
    if (typeof first === 'object' && first != null && 'type' in first) {
      if ((first as { type: string }).type === 'tool_result') {
        return false;
      }
    }
  }

  // Skip local command stdout
  if (typeof content === 'string' && content.includes('<local-command-stdout>')) {
    return false;
  }

  // Skip empty content
  if (!content || (typeof content === 'string' && content.trim() === '')) {
    return false;
  }

  return true;
}

/**
 * Find the index of the last actual user message, skipping tool results,
 * skill injections, and empty messages.
 */
export function findLastUserMessageIndex(transcript: TranscriptEntry[]): number | null {
  for (let i = transcript.length - 1; i >= 0; i--) {
    if (isUserPrompt(transcript, i)) {
      return i;
    }
  }
  return null;
}

export interface StoppedTurn {
  /** Index of the assistant entry that carries the final message */
  finalIdx: number;
  /** Index where the next turn starts (exclusive end of this turn) */
  endIdx: number;
  /** Whether the final entry has a final stop_reason, so its usage is complete */
  isComplete: boolean;
}

/**
 * Locate the turn a Stop hook fired for. Input queued during the turn (a message
 * from another session, a task notification) can be appended at the moment of
 * the Stop, so "the entries after the last user prompt" may already belong to
 * the next turn. The final assistant message is matched by its text instead,
 * and searched only after the user entry that carries `promptId`, so an earlier
 * turn with the same short reply does not match. A queued message can carry the
 * same `promptId`, so the turn ends at the next user prompt after the final
 * message rather than at the next `promptId`.
 *
 * Returns null while the final message is not in the transcript yet.
 */
export function findStoppedTurn(
  transcript: TranscriptEntry[],
  lastAssistantMessage: string,
  promptId?: string,
): StoppedTurn | null {
  const wanted = lastAssistantMessage.trim();
  if (!wanted) {
    return null;
  }

  let startIdx = 0;
  if (promptId) {
    startIdx = transcript.findIndex(
      (entry) => entry.type === 'user' && entry.promptId === promptId,
    );
    if (startIdx === -1) {
      return null;
    }
  }

  for (let i = startIdx; i < transcript.length; i++) {
    const entry = transcript[i];
    if (
      entry.type !== 'assistant' ||
      extractTextContent(entry.message?.content ?? '').trim() !== wanted
    ) {
      continue;
    }
    let endIdx = transcript.length;
    for (let j = i + 1; j < transcript.length; j++) {
      if (isUserPrompt(transcript, j)) {
        endIdx = j;
        break;
      }
    }
    const stopReason = entry.message?.stop_reason;
    return { finalIdx: i, endIdx, isComplete: stopReason != null && stopReason !== 'tool_use' };
  }
  return null;
}

/**
 * Find the final text response from the assistant after the given index.
 */
export function findFinalAssistantResponse(
  transcript: TranscriptEntry[],
  startIdx: number,
): string | null {
  let finalResponse: string | null = null;

  for (let i = startIdx; i < transcript.length; i++) {
    const entry = transcript[i];
    if (entry.type !== 'assistant') {
      continue;
    }

    const content = entry.message?.content;
    if (!Array.isArray(content)) {
      continue;
    }

    for (const part of content) {
      if (
        typeof part === 'object' &&
        part != null &&
        'type' in part &&
        part.type === 'text' &&
        'text' in part
      ) {
        const text = (part as { type: 'text'; text: string }).text;
        if (text.trim()) {
          finalResponse = text;
        }
      }
    }
  }

  return finalResponse;
}

/**
 * Get the timestamp (in ns) of the next transcript entry that has one.
 */
export function getNextTimestampNs(
  transcript: TranscriptEntry[],
  currentIdx: number,
): number | null {
  for (let i = currentIdx + 1; i < transcript.length; i++) {
    const ts = transcript[i].timestamp;
    if (ts) {
      return parseTimestampToNs(ts);
    }
  }
  return null;
}

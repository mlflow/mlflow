/**
 * Transcript parsing utilities for Codex CLI rollout JSONL files.
 *
 * Codex CLI transcripts use a RolloutLine format defined in
 * codex-rs/protocol/src/protocol.rs. Each line is:
 *   {"timestamp": "...", "type": "<variant>", "payload": {...}}
 *
 * Turns are delimited by event_msg task_started / task_complete pairs.
 *
 * References:
 * - Protocol types: github.com/openai/codex codex-rs/protocol/src/protocol.rs
 * - Rollout recorder: github.com/openai/codex codex-rs/rollout/src/recorder.rs
 */

import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import type {
  RolloutLine,
  ResponseItemPayload,
  EventMsgPayload,
  SessionMetaPayload,
  TokenUsage,
  ContentBlock,
} from './types.js';

const NANOSECONDS_PER_S = 1e9;
const NANOSECONDS_PER_MS = 1e6;

/**
 * Read and parse a Codex JSONL transcript file.
 */
export function readTranscript(path: string): RolloutLine[] {
  const content = readFileSync(path, 'utf-8');
  return content
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as RolloutLine);
}

/**
 * Parse an ISO timestamp string to nanoseconds since Unix epoch.
 */
export function parseTimestampToNs(timestamp: string | undefined | null): number | null {
  if (!timestamp) return null;
  try {
    const ms = new Date(timestamp).getTime();
    if (isNaN(ms)) return null;
    return ms * NANOSECONDS_PER_MS;
  } catch {
    return null;
  }
}

/**
 * Extract text from a response_item content field.
 * Content is an array of ContentBlock objects with type "input_text" or "output_text".
 */
export function extractTextFromContent(content: ContentBlock[] | string | undefined): string {
  if (!content) return '';
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .filter((block) => block.type === 'input_text' || block.type === 'output_text')
    .map((block) => block.text)
    .join('\n');
}

/**
 * Find the last user prompt in the transcript.
 * User prompts are response_item records with payload.type=message and payload.role=user
 * whose content has input_text blocks that aren't system/developer injections.
 */
export function findLastUserPrompt(records: RolloutLine[]): { text: string; index: number } | null {
  for (let i = records.length - 1; i >= 0; i--) {
    const record = records[i];
    if (record.type !== 'response_item') continue;
    const payload = record.payload as ResponseItemPayload;
    if (payload.type !== 'message' || payload.role !== 'user') continue;

    const text = extractTextFromContent(payload.content);
    // Skip system/developer context injections (start with XML-like tags)
    if (text && !text.startsWith('<')) {
      return { text, index: i };
    }
  }
  return null;
}

/**
 * Extract records belonging to the last turn.
 * Turns are delimited by event_msg records with type=task_started / task_complete.
 */
export function getLastTurnRecords(records: RolloutLine[]): RolloutLine[] {
  let lastStart: number | null = null;
  let lastEnd: number | null = null;

  for (let i = 0; i < records.length; i++) {
    if (records[i].type !== 'event_msg') continue;
    const payload = records[i].payload as EventMsgPayload;
    if (payload.type === 'task_started') lastStart = i;
    else if (payload.type === 'task_complete') lastEnd = i;
  }

  if (lastStart !== null) {
    const end = (lastEnd ?? records.length - 1) + 1;
    return records.slice(lastStart, end);
  }
  return records;
}

/**
 * Extract cumulative token usage from the last token_count event in a set of records.
 */
export function getTokenUsage(records: RolloutLine[]): TokenUsage | null {
  let usage: TokenUsage | null = null;
  for (const record of records) {
    if (record.type !== 'event_msg') continue;
    const payload = record.payload as EventMsgPayload;
    if (payload.type !== 'token_count') continue;
    if (payload.info?.last_token_usage) {
      usage = payload.info.last_token_usage;
    }
  }
  return usage;
}

/**
 * Extract model name from session_meta or turn_context records.
 */
export function getModel(records: RolloutLine[]): string {
  for (const record of records) {
    if (record.type === 'session_meta' || record.type === 'turn_context') {
      const model = (record.payload as Record<string, unknown>).model;
      if (typeof model === 'string') return model;
    }
  }
  return 'unknown';
}

/**
 * Extract session ID from the session_meta record.
 */
export function getSessionId(records: RolloutLine[]): string | null {
  for (const record of records) {
    if (record.type === 'session_meta') {
      return (record.payload as SessionMetaPayload).id ?? null;
    }
  }
  return null;
}

/**
 * Build a map from function_call call_id to function_call_output output.
 */
export function buildToolResultMap(records: RolloutLine[]): Record<string, string> {
  const results: Record<string, string> = {};
  for (const record of records) {
    if (record.type !== 'response_item') continue;
    const payload = record.payload as ResponseItemPayload;
    if (payload.type === 'function_call_output' && payload.call_id) {
      results[payload.call_id] = payload.output ?? '';
    }
  }
  return results;
}

/**
 * Find the transcript rollout file for a given thread ID.
 *
 * Codex stores transcripts at:
 *   ~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<thread-id>.jsonl
 *
 * This is optional enrichment — if not found, tracing still works
 * from the notify payload alone.
 */
export function findTranscriptForThread(threadId: string): string | null {
  try {
    const sessionsDir = join(homedir(), '.codex', 'sessions');
    if (!existsSync(sessionsDir)) return null;

    // Walk year/month/day directories in reverse to find the most recent
    const years = readdirSync(sessionsDir).sort().reverse();
    for (const year of years) {
      const yearDir = join(sessionsDir, year);
      const months = readdirSync(yearDir).sort().reverse();
      for (const month of months) {
        const monthDir = join(yearDir, month);
        const days = readdirSync(monthDir).sort().reverse();
        for (const day of days) {
          const dayDir = join(monthDir, day);
          const files = readdirSync(dayDir).filter(
            (f) => f.endsWith('.jsonl') && f.includes(threadId),
          );
          if (files.length > 0) {
            return join(dayDir, files[0]);
          }
        }
      }
    }
  } catch {
    // Transcript lookup is best-effort
  }
  return null;
}

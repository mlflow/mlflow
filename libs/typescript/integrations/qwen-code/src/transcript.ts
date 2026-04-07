/**
 * Transcript parsing utilities for Qwen Code JSONL files.
 *
 * Qwen Code transcripts are JSONL files with tree-structured ChatRecords.
 * Each record has uuid/parentUuid forming a parent-child tree.
 * Messages use Gemini-style format: {role, parts: [{text}]}.
 *
 * Location: ~/.qwen/projects/<project-id>/chats/<sessionId>.jsonl
 */

import { readFileSync } from 'node:fs';
import type { ChatRecord, GeminiMessage, UsageMetadata } from './types.js';

const NANOSECONDS_PER_MS = 1e6;

/**
 * Read and parse a Qwen Code JSONL transcript file.
 */
export function readTranscript(path: string): ChatRecord[] {
  const content = readFileSync(path, 'utf-8');
  return content
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as ChatRecord);
}

/**
 * Parse an ISO timestamp string to nanoseconds since Unix epoch.
 */
export function parseTimestampToNs(timestamp: string | undefined | null): number | null {
  if (!timestamp) {
    return null;
  }
  try {
    const ms = new Date(timestamp).getTime();
    if (isNaN(ms)) {
      return null;
    }
    return ms * NANOSECONDS_PER_MS;
  } catch {
    return null;
  }
}

/**
 * Extract text from a ChatRecord's message field.
 *
 * Qwen Code messages use Gemini-style format: {role, parts: [{text}]}.
 * Also handles plain strings for flexibility.
 */
export function getMessageText(record: ChatRecord): string {
  const msg = record.message;
  if (typeof msg === 'string') {
    return msg;
  }
  if (isGeminiMessage(msg)) {
    return msg.parts.map((p) => p.text).join('\n');
  }
  return String(msg);
}

function isGeminiMessage(msg: unknown): msg is GeminiMessage {
  return (
    typeof msg === 'object' &&
    msg != null &&
    'parts' in msg &&
    Array.isArray((msg as GeminiMessage).parts)
  );
}

/**
 * Find the last user record in the transcript.
 */
export function findLastUserRecord(records: ChatRecord[]): ChatRecord | null {
  for (let i = records.length - 1; i >= 0; i--) {
    if (records[i].type === 'user') {
      const text = getMessageText(records[i]);
      if (text.trim()) {
        return records[i];
      }
    }
  }
  return null;
}

/**
 * Build parent-child maps from the ChatRecord tree.
 */
export function buildRecordTree(records: ChatRecord[]): {
  byUuid: Map<string, ChatRecord>;
  children: Map<string, string[]>;
} {
  const byUuid = new Map<string, ChatRecord>();
  const children = new Map<string, string[]>();

  for (const record of records) {
    byUuid.set(record.uuid, record);
    if (!children.has(record.uuid)) {
      children.set(record.uuid, []);
    }
    if (record.parentUuid) {
      if (!children.has(record.parentUuid)) {
        children.set(record.parentUuid, []);
      }
      children.get(record.parentUuid)?.push(record.uuid);
    }
  }

  return { byUuid, children };
}

/**
 * Extract token usage from a ChatRecord's usageMetadata.
 */
export function getTokenUsage(
  metadata: UsageMetadata | undefined,
): { input: number; output: number; total: number } | null {
  if (!metadata) {
    return null;
  }

  const input = metadata.promptTokenCount ?? metadata.input_tokens ?? 0;
  const output = metadata.candidatesTokenCount ?? metadata.output_tokens ?? 0;
  const total = metadata.totalTokenCount ?? input + output;

  if (input === 0 && output === 0) {
    return null;
  }
  return { input, output, total };
}

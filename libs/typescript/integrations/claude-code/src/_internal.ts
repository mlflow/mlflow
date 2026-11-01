/**
 * Internal helpers shared between the CLI transcript path (tracing.ts) and the
 * live SDK-stream path (liveTracing.ts). Not part of the package's public API:
 * the leading underscore signals "do not import from `@mlflow/claude-code`".
 */

import { TokenUsageKey } from '@mlflow/core';

import type { ContentBlock, MessageContent, TokenUsage } from './types.js';

// ============================================================================
// Shared constants
// ============================================================================

export const MAX_PREVIEW_LENGTH = 1000;

export const METADATA_KEY_CLAUDE_CODE_VERSION = 'mlflow.claude_code_version';
export const METADATA_KEY_WORKING_DIRECTORY = 'mlflow.trace.working_directory';
export const METADATA_KEY_PERMISSION_MODE = 'mlflow.trace.permission_mode';

export function getCurrentUser(): string {
  return process.env.USER || process.env.USERNAME || '';
}

// ============================================================================
// Content / token helpers
// ============================================================================

/**
 * Separate text content from tool_use blocks in an assistant response.
 */
export function extractContentAndTools(
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

/**
 * Claude Code can emit several content-block records for one API response.
 * Their usage counters describe the same response and must not be summed.
 */
export function mergeAssistantMessage(
  previous: MessageContent,
  current: MessageContent,
): MessageContent {
  const blocks = (content: MessageContent['content']): ContentBlock[] =>
    typeof content === 'string'
      ? [{ type: 'text', text: content }]
      : Array.isArray(content)
        ? content
        : [];
  const content = [...blocks(previous.content)];
  for (const block of blocks(current.content)) {
    const toolIndex =
      block.type === 'tool_use'
        ? content.findIndex((part) => part.type === 'tool_use' && part.id === block.id)
        : -1;
    if (toolIndex >= 0) {
      content[toolIndex] = block;
    } else if (!content.some((part) => JSON.stringify(part) === JSON.stringify(block))) {
      content.push(block);
    }
  }

  let usage = previous.usage ?? current.usage;
  if (previous.usage && current.usage) {
    usage = { ...previous.usage, ...current.usage };
    for (const key of [
      'input_tokens',
      'output_tokens',
      'cache_read_input_tokens',
      'cache_creation_input_tokens',
    ] as const) {
      if (previous.usage[key] !== undefined || current.usage[key] !== undefined) {
        usage[key] = Math.max(previous.usage[key] ?? 0, current.usage[key] ?? 0);
      }
    }
  }
  return { ...previous, ...current, model: current.model ?? previous.model, content, usage };
}

/**
 * Normalize a Claude Code usage payload into the TOKEN_USAGE schema.
 * Includes cache-read and cache-creation tokens in `input_tokens` and
 * `total_tokens`, matching `mlflow.anthropic.autolog`. Cache tokens are also
 * exposed as separate optional keys so consumers can compute cache hit rate.
 */
export function buildUsageDict(usage: TokenUsage): Record<string, number> {
  // Anthropic reports input_tokens excluding cache tokens.
  const inputTokens =
    (usage.input_tokens ?? 0) +
    (usage.cache_read_input_tokens ?? 0) +
    (usage.cache_creation_input_tokens ?? 0);
  const outputTokens = usage.output_tokens ?? 0;

  const usageDict: Record<string, number> = {
    [TokenUsageKey.INPUT_TOKENS]: inputTokens,
    [TokenUsageKey.OUTPUT_TOKENS]: outputTokens,
    [TokenUsageKey.TOTAL_TOKENS]: inputTokens + outputTokens,
  };
  if (usage.cache_read_input_tokens !== undefined) {
    usageDict[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = usage.cache_read_input_tokens;
  }
  if (usage.cache_creation_input_tokens !== undefined) {
    usageDict[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] = usage.cache_creation_input_tokens;
  }
  return usageDict;
}

// ============================================================================
// JSON-safe sanitization
// ============================================================================

/**
 * Deep-clone a value, replacing functions with a placeholder. Used to record
 * the caller's `options` on the root span without serializing callbacks like
 * `canUseTool`, `mcpServers[].command`, or `hooks[].hooks[]`. Returning the
 * original `options` object would leak function references into the span
 * inputs (which then fail to JSON-serialize or grow unbounded).
 */
export function sanitizeForSpan(value: unknown, seen: WeakSet<object> = new WeakSet()): unknown {
  if (typeof value === 'function') {
    return '[function]';
  }
  if (value == null || typeof value !== 'object') {
    return value;
  }
  if (seen.has(value)) {
    return '[circular]';
  }
  seen.add(value);
  if (Array.isArray(value)) {
    return value.map((v) => sanitizeForSpan(v, seen));
  }
  const result: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) {
    result[k] = sanitizeForSpan(v, seen);
  }
  return result;
}

/**
 * Internal helpers shared between the CLI transcript path (tracing.ts) and the
 * live SDK-stream path (liveTracing.ts). Not part of the package's public API:
 * the leading underscore signals "do not import from `@mlflow/claude-code`".
 */

import { TokenUsageKey } from '@mlflow/core';

import type { ContentBlock, TokenUsage } from './types.js';

// ============================================================================
// Shared constants
// ============================================================================

export const MAX_PREVIEW_LENGTH = 1000;

export const METADATA_KEY_CLAUDE_CODE_VERSION = 'mlflow.claude_code_version';
export const METADATA_KEY_WORKING_DIRECTORY = 'mlflow.trace.working_directory';
export const METADATA_KEY_PERMISSION_MODE = 'mlflow.trace.permission_mode';

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
 * Normalize a Claude Code usage payload into the TOKEN_USAGE schema.
 * Stores fields as the Anthropic API reports them, matching
 * `mlflow.anthropic.autolog`: `input_tokens` is the non-cached input,
 * cache tokens are exposed as separate optional keys so consumers can
 * compute cache hit rate, and `total_tokens` follows the
 * `mlflow.anthropic` convention of `input_tokens + output_tokens`
 * (cache tokens excluded).
 */
export function buildUsageDict(usage: TokenUsage): Record<string, number> {
  const inputTokens = usage.input_tokens ?? 0;
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

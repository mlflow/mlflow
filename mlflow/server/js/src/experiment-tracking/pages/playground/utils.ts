import { uniq } from 'lodash';
import type { ChatMessage } from './types';

// Mirrors the canonical `PROMPT_TEMPLATE_VARIABLE_PATTERN` in
// `mlflow/prompt/constants.py`. The flag is `g` so we can `matchAll` across an
// entire message body and `replaceAll`-style across substitution sites.
const TEMPLATE_VARIABLE_PATTERN = /\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}/g;

/**
 * Returns the unique `{{ var }}` placeholders that appear anywhere in the
 * messages, in order of first appearance.
 */
export const extractTemplateVariables = (messages: ChatMessage[]): string[] => {
  const names: string[] = [];
  for (const message of messages) {
    const matches = (message.content ?? '').matchAll(TEMPLATE_VARIABLE_PATTERN);
    for (const match of matches) {
      names.push(match[1]);
    }
  }
  return uniq(names);
};

/**
 * Returns a new messages array where each `{{ var }}` is replaced with
 * `values[var]` (empty string if missing). Roles, order, and any placeholders
 * the strict regex does not match (e.g. malformed `{{ }}`) are preserved
 * literally.
 */
export const substituteVariables = (messages: ChatMessage[], values: Record<string, string>): ChatMessage[] => {
  return messages.map((message) => ({
    ...message,
    content: (message.content ?? '').replace(TEMPLATE_VARIABLE_PATTERN, (_, name: string) => values[name] ?? ''),
  }));
};

/**
 * Returns the unique declared variable names whose corresponding value is
 * missing or trims to an empty string. Used to surface a pre-submit warning
 * when placeholders would silently substitute as `''`.
 */
export const getEmptyVariables = (messages: ChatMessage[], values: Record<string, string>): string[] => {
  return extractTemplateVariables(messages).filter((name) => !(values[name] ?? '').trim());
};

/**
 * Pretty-prints a JSON string with 2-space indentation. Falls back to the
 * original string when it is not valid JSON (e.g. a partial or malformed
 * tool-call argument), so the displayed text stays verbatim rather than being
 * dropped. Used by `JsonCodeBlock` to render assistant tool-call arguments and
 * JSON-format responses.
 */
export const prettyPrintJson = (raw: string): string => {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
};

/**
 * Returns true when the JSON tool-definitions text has no usable tools — either
 * empty/whitespace or parses to an empty array. Parse errors return false so
 * they flow to the separate parse-error path.
 */
export const isToolsValueEmpty = (text: string): boolean => {
  if (!text.trim()) {
    return true;
  }
  try {
    const parsed = JSON.parse(text);
    return Array.isArray(parsed) && parsed.length === 0;
  } catch {
    return false;
  }
};

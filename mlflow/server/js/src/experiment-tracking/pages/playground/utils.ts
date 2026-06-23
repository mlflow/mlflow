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
 * Bare-minimum JSON Schema used to pre-populate a freshly added tool's
 * parameters editor and the response-format schema editor, mirroring the
 * "blank" template in the design spec.
 */
export const BLANK_JSON_SCHEMA = `{
  "type": "object",
  "properties": {},
  "required": []
}`;

/**
 * Pretty-prints JSON text with 2-space indentation. Returns `null` when the text
 * is not valid JSON so callers can leave the input untouched (and keep showing the
 * parse error). Used by the editor's Format action.
 */
export const formatJson = (text: string): string | null => {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return null;
  }
};

/**
 * Validates a tool's parameters JSON Schema text. Returns `null` when it parses
 * to a JSON object, or a short reason otherwise: empty, a JSON parse error, or a
 * non-object value. Used to flag the parameters editor and gate submission.
 */
export const getToolParametersError = (text: string): string | null => {
  if (!text.trim()) {
    return 'Add a parameters schema';
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    return e instanceof Error ? e.message : 'Invalid JSON';
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return 'Parameters must be a JSON object';
  }
  return null;
};

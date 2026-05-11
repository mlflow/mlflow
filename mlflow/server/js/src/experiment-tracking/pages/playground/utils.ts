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
    const matches = message.content.matchAll(TEMPLATE_VARIABLE_PATTERN);
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
    content: message.content.replace(TEMPLATE_VARIABLE_PATTERN, (_, name: string) => values[name] ?? ''),
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

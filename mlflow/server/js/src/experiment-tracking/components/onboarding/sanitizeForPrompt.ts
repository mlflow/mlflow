/**
 * Sanitize a user-controlled value before interpolating it into a coding-agent
 * prompt. The receiving agent (Claude Code, Cursor, etc.) treats the entire
 * prompt as one untrusted text blob; without sanitization, a maliciously named
 * experiment could inject its own "INSTRUCTIONS:" block on a new line and the
 * agent may obey.
 *
 * We strip CR/LF + other ASCII control chars (collapses multi-line injection
 * to a single line), neutralize backticks and double-quotes (the delimiters our
 * prompts use), and cap length to bound the injection payload size.
 */
export const sanitizeForPrompt = (value: string, maxLength = 200): string => {
  // eslint-disable-next-line no-control-regex -- intentionally matching control chars
  const oneLine = value.replace(/[\x00-\x1F\x7F]/g, ' ');
  const safeQuotes = oneLine.replace(/[`"]/g, "'");
  const trimmed = safeQuotes.trim();
  return trimmed.length > maxLength ? trimmed.slice(0, maxLength) : trimmed;
};

// JSON accepts only these four whitespace characters, including before its first token.
const JSON_PREFIX = /^[ \t\r\n]*(?:[[{"0-9-]|(?:true|false|null)[ \t\r\n]*$)/;

export const mayBeJsonText = (value: unknown): value is string => typeof value === 'string' && JSON_PREFIX.test(value);

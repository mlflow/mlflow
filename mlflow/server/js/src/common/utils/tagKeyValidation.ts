/**
 * Tag key validation aligned with MLflow backend (utils/validation.py).
 * Backend allows: alphanumerics, underscores (_), dashes (-), periods (.),
 * spaces ( ), colons (:) and slashes (/). Keys must not start with "/",
 * must not be "." or start with "..".
 * Path-like patterns (e.g. "/.", "/..", "//") are not validated here; the backend
 * will reject invalid keys and the UI displays the backend error message.
 */

/**
 * Allowed characters for tag keys (matches backend validate_param_and_metric_name).
 * Backend regex: ^[/\w.\- :]*$ (non-Windows)
 */
const TAG_KEY_ALLOWED_CHARS_REGEX = /^[\w.\- :/]*$/;

/**
 * Returns true if the tag key passes basic character and path rules.
 * Stricter path rules (e.g. path traversal) are enforced by the backend; the UI
 * shows the backend error when save fails.
 */
export function isValidTagKey(key: string): boolean {
  if (key === '') return true;
  if (!TAG_KEY_ALLOWED_CHARS_REGEX.test(key)) return false;
  if (key.startsWith('/')) return false;
  if (key === '.') return false;
  if (key.startsWith('..')) return false;
  return true;
}

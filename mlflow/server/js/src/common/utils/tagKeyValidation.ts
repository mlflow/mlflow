/**
 * Tag key validation aligned with MLflow backend (utils/validation.py).
 * Backend allows: alphanumerics, underscores (_), dashes (-), periods (.),
 * spaces ( ), colons (:) and slashes (/). Keys must not start with "/",
 * must not be "." or start with "..", and must not contain path traversal (e.g. "/." or "/..").
 */

/**
 * Allowed characters for tag keys (matches backend validate_param_and_metric_name).
 * Backend regex: ^[/\w.\- :]*$ (non-Windows)
 */
const TAG_KEY_ALLOWED_CHARS_REGEX = /^[\w.\- :/]*$/;

/**
 * Returns true if the tag key is valid (allowed characters and path rules).
 * Mirrors backend _validate_tag_name + path_not_unique logic.
 */
export function isValidTagKey(key: string): boolean {
  if (key === '') return true;
  if (!TAG_KEY_ALLOWED_CHARS_REGEX.test(key)) return false;
  // path_not_unique: no leading slash, not ".", not "..", no path traversal
  if (key.startsWith('/')) return false;
  if (key === '.') return false;
  if (key.startsWith('..')) return false;
  if (key.includes('/.') || key.includes('/..') || key.includes('//')) return false;
  return true;
}

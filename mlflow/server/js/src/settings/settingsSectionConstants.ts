/** URL path segment for Settings > LLM Connections (`/settings/llm-connections`). */
export const SETTINGS_SECTION_LLM_CONNECTIONS = 'llm-connections';

/** Allowed path segments for `/settings/:section`. */
export const SETTINGS_PATH_SEGMENTS = ['general', SETTINGS_SECTION_LLM_CONNECTIONS, 'webhooks'] as const;

export type SettingsPathSegment = (typeof SETTINGS_PATH_SEGMENTS)[number];

export function isSettingsPathSegment(value: string): value is SettingsPathSegment {
  return (SETTINGS_PATH_SEGMENTS as readonly string[]).includes(value);
}

/** Query param carrying the path to return to when leaving the Settings sub-sidebar. */
export const SETTINGS_RETURN_TO_PARAM = 'returnTo';

/** Returns a safe in-app location string; invalid or settings URLs fall back to `fallback`. */
export function sanitizeSettingsReturnPath(value: string | undefined, fallback: string): string {
  if (!value || typeof value !== 'string') {
    return fallback;
  }
  if (!value.startsWith('/') || value.startsWith('//')) {
    return fallback;
  }
  // Parse to get just the pathname, so /settings?workspace=... and /settings#hash are also rejected.
  try {
    const { pathname } = new URL(value, 'http://localhost');
    if (pathname === '/settings' || pathname.startsWith('/settings/')) {
      return fallback;
    }
  } catch {
    return fallback;
  }
  return value;
}

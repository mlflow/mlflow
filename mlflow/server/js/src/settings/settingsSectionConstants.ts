/** URL path segment for Settings > General (`/settings/general`). */
export const SETTINGS_SECTION_GENERAL = 'general';

/** URL path segment for Settings > LLM Connections (`/settings/llm-connections`). */
export const SETTINGS_SECTION_LLM_CONNECTIONS = 'llm-connections';

/** URL path segment for Settings > Webhooks (`/settings/webhooks`). */
export const SETTINGS_SECTION_WEBHOOKS = 'webhooks';

/** Allowed path segments for `/settings/:section`. */
export const SETTINGS_PATH_SEGMENTS = [
  SETTINGS_SECTION_GENERAL,
  SETTINGS_SECTION_LLM_CONNECTIONS,
  SETTINGS_SECTION_WEBHOOKS,
] as const;

export type SettingsPathSegment = (typeof SETTINGS_PATH_SEGMENTS)[number];

export function isSettingsPathSegment(value: string): value is SettingsPathSegment {
  return (SETTINGS_PATH_SEGMENTS as readonly string[]).includes(value);
}

/** Query param carrying the path to return to when leaving the Settings sub-sidebar. */
export const SETTINGS_RETURN_TO_PARAM = 'returnTo';

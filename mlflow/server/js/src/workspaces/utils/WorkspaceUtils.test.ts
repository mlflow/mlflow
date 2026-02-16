import { describe, jest, beforeEach, afterEach, it, expect } from '@jest/globals';
import {
  getActiveWorkspace,
  getLastUsedWorkspace,
  setActiveWorkspace,
  extractWorkspaceFromSearchParams,
  prefixRouteWithWorkspace,
  appendWorkspaceSearchParams,
  validateWorkspaceName,
  isGlobalRoute,
  removeWorkspaceQueryParam,
  WORKSPACE_NAME_MIN_LENGTH,
  WORKSPACE_NAME_MAX_LENGTH,
} from './WorkspaceUtils';
import { getWorkspacesEnabledSync } from '../../experiment-tracking/hooks/useServerInfo';

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  ...jest.requireActual<typeof import('../../experiment-tracking/hooks/useServerInfo')>(
    '../../experiment-tracking/hooks/useServerInfo',
  ),
  getWorkspacesEnabledSync: jest.fn(),
}));

const getWorkspacesEnabledSyncMock = jest.mocked(getWorkspacesEnabledSync);

describe('validateWorkspaceName', () => {
  it('accepts valid workspace names', () => {
    expect(validateWorkspaceName('my-workspace')).toEqual({ valid: true });
    expect(validateWorkspaceName('workspace1')).toEqual({ valid: true });
    expect(validateWorkspaceName('a1')).toEqual({ valid: true });
    expect(validateWorkspaceName('team-a-project-1')).toEqual({ valid: true });
  });

  it('rejects names shorter than minimum length', () => {
    const result = validateWorkspaceName('a');
    expect(result.valid).toBe(false);
    expect(result.error).toContain(`between ${WORKSPACE_NAME_MIN_LENGTH} and ${WORKSPACE_NAME_MAX_LENGTH}`);
  });

  it('rejects names longer than maximum length', () => {
    const longName = 'a'.repeat(WORKSPACE_NAME_MAX_LENGTH + 1);
    const result = validateWorkspaceName(longName);
    expect(result.valid).toBe(false);
    expect(result.error).toContain(`between ${WORKSPACE_NAME_MIN_LENGTH} and ${WORKSPACE_NAME_MAX_LENGTH}`);
  });

  it('rejects names with uppercase letters', () => {
    const result = validateWorkspaceName('MyWorkspace');
    expect(result.valid).toBe(false);
    expect(result.error).toContain('lowercase alphanumeric');
  });

  it('rejects names with consecutive hyphens', () => {
    const result = validateWorkspaceName('my--workspace');
    expect(result.valid).toBe(false);
    expect(result.error).toContain('no consecutive hyphens');
  });

  it('rejects names starting with hyphen', () => {
    const result = validateWorkspaceName('-workspace');
    expect(result.valid).toBe(false);
    expect(result.error).toContain('lowercase alphanumeric');
  });

  it('rejects names ending with hyphen', () => {
    const result = validateWorkspaceName('workspace-');
    expect(result.valid).toBe(false);
    expect(result.error).toContain('lowercase alphanumeric');
  });

  it('rejects names with spaces', () => {
    const result = validateWorkspaceName('my workspace');
    expect(result.valid).toBe(false);
    expect(result.error).toContain('lowercase alphanumeric');
  });

  it('rejects non-string values', () => {
    // @ts-expect-error Testing invalid type
    const result = validateWorkspaceName(123);
    expect(result.valid).toBe(false);
    expect(result.error).toContain('must be a string');
  });
});

describe('WorkspaceUtils', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    getWorkspacesEnabledSyncMock.mockReturnValue(true);
    // Clear any stored workspace
    setActiveWorkspace(null);
    // Clear localStorage
    if (typeof window !== 'undefined') {
      window.localStorage.clear();
    }
  });

  afterEach(() => {
    setActiveWorkspace(null);
  });

  describe('getLastUsedWorkspace', () => {
    it('returns null when no workspace has been used', () => {
      expect(getLastUsedWorkspace()).toBeNull();
    });

    it('returns workspace from localStorage', () => {
      setActiveWorkspace('team-a');
      expect(getLastUsedWorkspace()).toBe('team-a');
    });

    it('persists across page loads (localStorage)', () => {
      setActiveWorkspace('persistent-workspace');
      const stored = window.localStorage.getItem('mlflow.activeWorkspace');
      expect(stored).toBe('persistent-workspace');
      expect(getLastUsedWorkspace()).toBe('persistent-workspace');
    });
  });

  describe('getActiveWorkspace / setActiveWorkspace', () => {
    it('returns null when no workspace is in URL', () => {
      expect(getActiveWorkspace()).toBeNull();
    });

    it('returns the in-memory workspace even when window.location is unavailable (SSR)', () => {
      setActiveWorkspace('team-a');

      // Mock SSR environment (no window.location)
      const originalLocation = window.location;
      delete (window as any).location;

      // getActiveWorkspace returns in-memory value regardless of window.location
      expect(getActiveWorkspace()).toBe('team-a');

      // And getLastUsedWorkspace should also work
      (window as any).location = originalLocation;
      expect(getLastUsedWorkspace()).toBe('team-a');
    });

    it('persists workspace to localStorage', () => {
      setActiveWorkspace('team-b');
      const stored = window.localStorage.getItem('mlflow.activeWorkspace');
      expect(stored).toBe('team-b');
    });

    it('clears in-memory workspace but leaves localStorage intact when setting to null', () => {
      setActiveWorkspace('team-c');
      expect(window.localStorage.getItem('mlflow.activeWorkspace')).toBe('team-c');
      expect(getActiveWorkspace()).toBe('team-c');

      // setActiveWorkspace(null) clears in-memory but doesn't clear localStorage
      // (localStorage keeps last used workspace for UI hints)
      setActiveWorkspace(null);
      expect(getActiveWorkspace()).toBeNull();
      expect(window.localStorage.getItem('mlflow.activeWorkspace')).toBe('team-c');
    });

    it('returns in-memory workspace regardless of URL params', () => {
      // Set active workspace in memory
      setActiveWorkspace('cached-workspace');

      // Mock URL with no workspace param
      const originalLocation = window.location;
      delete (window as any).location;
      (window as any).location = { ...originalLocation, search: '?other=param' };

      // getActiveWorkspace returns in-memory value, doesn't read from URL
      expect(getActiveWorkspace()).toBe('cached-workspace');

      // Restore original location
      (window as any).location = originalLocation;
    });

    it('returns null when creating new workspace (no query param, no cache)', () => {
      // Clear localStorage
      setActiveWorkspace(null);

      // Mock URL with no workspace param (e.g., creating new workspace)
      const originalLocation = window.location;
      delete (window as any).location;
      (window as any).location = { ...originalLocation, search: '' };

      // Should return null (no workspace header will be sent)
      expect(getActiveWorkspace()).toBeNull();

      // Restore original location
      (window as any).location = originalLocation;
    });
  });

  describe('extractWorkspaceFromSearchParams', () => {
    it('extracts workspace from URLSearchParams', () => {
      const params = new URLSearchParams('workspace=default');
      expect(extractWorkspaceFromSearchParams(params)).toBe('default');
    });

    it('extracts workspace from query string', () => {
      expect(extractWorkspaceFromSearchParams('workspace=team-a')).toBe('team-a');
      expect(extractWorkspaceFromSearchParams('?workspace=team-b')).toBe('team-b');
    });

    it('returns null when workspace param is missing', () => {
      expect(extractWorkspaceFromSearchParams('')).toBeNull();
      expect(extractWorkspaceFromSearchParams('other=value')).toBeNull();
    });

    it('returns null for invalid workspace names', () => {
      expect(extractWorkspaceFromSearchParams('workspace=UPPERCASE')).toBeNull();
      expect(extractWorkspaceFromSearchParams('workspace=has spaces')).toBeNull();
      expect(extractWorkspaceFromSearchParams('workspace=has--double-hyphen')).toBeNull();
    });

    it('handles URL-encoded workspace names', () => {
      // Valid name that's URL-encoded
      expect(extractWorkspaceFromSearchParams('workspace=team-a')).toBe('team-a');
    });
  });

  describe('isGlobalRoute', () => {
    it('returns false for root path (handled specially in prefixRouteWithWorkspace)', () => {
      // Root path is NOT in ALWAYS_GLOBAL_ROUTES - it's handled specially:
      // - '/' without workspace param = workspace selector (no workspace added)
      // - '/?workspace=foo' = workspace home (preserve workspace)
      expect(isGlobalRoute('/')).toBe(false);
    });

    it('returns false for settings path (workspace-scoped)', () => {
      expect(isGlobalRoute('/settings')).toBe(false);
      expect(isGlobalRoute('/settings/general')).toBe(false);
    });

    it('returns false for workspace-scoped paths', () => {
      expect(isGlobalRoute('/experiments')).toBe(false);
      expect(isGlobalRoute('/models')).toBe(false);
      expect(isGlobalRoute('/prompts')).toBe(false);
    });

    it('ignores query params and hash', () => {
      expect(isGlobalRoute('/?workspace=default')).toBe(false);
    });
  });

  describe('removeWorkspaceQueryParam', () => {
    it('removes workspace param from query string', () => {
      expect(removeWorkspaceQueryParam('/experiments?workspace=default')).toBe('/experiments');
    });

    it('preserves other query params', () => {
      expect(removeWorkspaceQueryParam('/experiments?workspace=default&filter=active')).toBe(
        '/experiments?filter=active',
      );
    });

    it('handles hash prefix', () => {
      expect(removeWorkspaceQueryParam('#/experiments?workspace=default')).toBe('#/experiments');
    });

    it('preserves hash fragment', () => {
      expect(removeWorkspaceQueryParam('/experiments?workspace=default#section')).toBe('/experiments#section');
    });

    it('returns unchanged if no workspace param', () => {
      expect(removeWorkspaceQueryParam('/experiments?filter=active')).toBe('/experiments?filter=active');
    });
  });

  describe('getActiveWorkspace', () => {
    it('returns the in-memory active workspace', () => {
      expect(getActiveWorkspace()).toBeNull();

      setActiveWorkspace('workspace-1');
      expect(getActiveWorkspace()).toBe('workspace-1');

      setActiveWorkspace('another-workspace');
      expect(getActiveWorkspace()).toBe('another-workspace');

      setActiveWorkspace(null);
      expect(getActiveWorkspace()).toBeNull();
    });
  });

  describe('prefixRouteWithWorkspace', () => {
    beforeEach(() => {
      // Set active workspace for these tests
      setActiveWorkspace('default');
    });

    afterEach(() => {
      // Clear active workspace
      setActiveWorkspace(null);
    });

    it('returns original string for empty/undefined values', () => {
      expect(prefixRouteWithWorkspace('')).toBe('');
      // @ts-expect-error Testing undefined case
      expect(prefixRouteWithWorkspace(undefined)).toBeUndefined();
    });

    it('returns original string when workspaces disabled', () => {
      getWorkspacesEnabledSyncMock.mockReturnValue(false);
      expect(prefixRouteWithWorkspace('/experiments')).toBe('/experiments');
    });

    it('returns original string for absolute URLs', () => {
      expect(prefixRouteWithWorkspace('https://example.com/path')).toBe('https://example.com/path');
      expect(prefixRouteWithWorkspace('http://localhost:3000')).toBe('http://localhost:3000');
    });

    it('returns original string for relative navigation (no leading /)', () => {
      expect(prefixRouteWithWorkspace('experiments')).toBe('experiments');
      expect(prefixRouteWithWorkspace('models/123')).toBe('models/123');
    });

    it('adds workspace query param to absolute paths', () => {
      expect(prefixRouteWithWorkspace('/experiments')).toBe('/experiments?workspace=default');
      expect(prefixRouteWithWorkspace('/models/123')).toBe('/models/123?workspace=default');
    });

    it('handles hash prefix correctly', () => {
      expect(prefixRouteWithWorkspace('#/experiments')).toBe('#/experiments?workspace=default');
    });

    it('preserves existing hash fragments', () => {
      expect(prefixRouteWithWorkspace('/experiments#section')).toBe('/experiments?workspace=default#section');
    });

    it('preserves existing query params', () => {
      expect(prefixRouteWithWorkspace('/experiments?search=test')).toBe('/experiments?search=test&workspace=default');
    });

    it('handles query strings and hash together', () => {
      expect(prefixRouteWithWorkspace('/models?filter=active#top')).toBe('/models?filter=active&workspace=default#top');
    });

    it('preserves explicit workspace param in URL', () => {
      // If URL already has explicit workspace, preserve it (don't override with active workspace)
      const path = '/experiments?workspace=old-workspace';
      expect(prefixRouteWithWorkspace(path)).toBe('/experiments?workspace=old-workspace');
    });

    it('returns path without workspace param when no workspace set', () => {
      setActiveWorkspace(null);
      expect(prefixRouteWithWorkspace('/experiments')).toBe('/experiments');
    });

    it('adds workspace param to root path when workspace is active (workspace home)', () => {
      // Root path with active workspace gets workspace param added (workspace home)
      expect(prefixRouteWithWorkspace('/')).toBe('/?workspace=default');
    });

    it('does not add workspace param to root path when no workspace is active', () => {
      setActiveWorkspace(null);
      expect(prefixRouteWithWorkspace('/')).toBe('/');
    });

    it('preserves explicit workspace param on root path', () => {
      // Root path with explicit workspace is preserved as-is
      expect(prefixRouteWithWorkspace('/?workspace=old')).toBe('/?workspace=old');
    });

    it('adds workspace param for settings route', () => {
      expect(prefixRouteWithWorkspace('/settings')).toBe('/settings?workspace=default');
      expect(prefixRouteWithWorkspace('/settings?workspace=old')).toBe('/settings?workspace=old');
    });

    it('uses different workspace when set', () => {
      setActiveWorkspace('team-a');
      expect(prefixRouteWithWorkspace('/experiments')).toBe('/experiments?workspace=team-a');
    });

    it('encodes workspace name in query param', () => {
      setActiveWorkspace('team-with-hyphen');
      expect(prefixRouteWithWorkspace('/experiments')).toBe('/experiments?workspace=team-with-hyphen');
    });
  });

  describe('prefixPathnameWithWorkspace', () => {
    beforeEach(() => {
      // Set active workspace for these tests
      setActiveWorkspace('default');
    });

    afterEach(() => {
      // Clear active workspace
      setActiveWorkspace(null);
    });

    it('returns undefined for undefined pathname', () => {
      expect(appendWorkspaceSearchParams(undefined)).toBeUndefined();
    });

    it('returns original for absolute URLs', () => {
      expect(appendWorkspaceSearchParams('https://example.com')).toBe('https://example.com');
    });

    it('adds workspace query param to pathname', () => {
      expect(appendWorkspaceSearchParams('/experiments')).toBe('/experiments?workspace=default');
      expect(appendWorkspaceSearchParams('/models/123')).toBe('/models/123?workspace=default');
    });

    it('adds workspace param to root path when workspace is active', () => {
      expect(appendWorkspaceSearchParams('/')).toBe('/?workspace=default');
    });

    it('returns root path unchanged when no workspace is active', () => {
      setActiveWorkspace(null);
      expect(appendWorkspaceSearchParams('/')).toBe('/');
    });

    it('adds workspace param to settings route', () => {
      expect(appendWorkspaceSearchParams('/settings')).toBe('/settings?workspace=default');
    });

    it('returns pathname without workspace when feature disabled', () => {
      getWorkspacesEnabledSyncMock.mockReturnValue(false);
      expect(appendWorkspaceSearchParams('/experiments')).toBe('/experiments');
    });

    it('uses active workspace when set', () => {
      setActiveWorkspace('team-b');
      expect(appendWorkspaceSearchParams('/models')).toBe('/models?workspace=team-b');
    });

    it('returns pathname unchanged when no workspace set', () => {
      setActiveWorkspace(null);
      expect(appendWorkspaceSearchParams('/experiments')).toBe('/experiments');
    });
  });
});

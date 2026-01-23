import { describe, jest, beforeEach, afterEach, it, expect } from '@jest/globals';
import {
  getActiveWorkspace,
  setActiveWorkspace,
  extractWorkspaceFromPathname,
  subscribeToWorkspaceChanges,
  setAvailableWorkspaces,
  getAvailableWorkspaces,
  prefixRouteWithWorkspace,
  prefixPathnameWithWorkspace,
  validateWorkspaceName,
  WORKSPACE_NAME_MIN_LENGTH,
  WORKSPACE_NAME_MAX_LENGTH,
} from './WorkspaceUtils';
import { getWorkspacesEnabledSync } from './ServerFeaturesContext';

jest.mock('./ServerFeaturesContext', () => ({
  ...jest.requireActual<typeof import('./ServerFeaturesContext')>('./ServerFeaturesContext'),
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
    setAvailableWorkspaces([]);
    // Clear localStorage
    if (typeof window !== 'undefined') {
      window.localStorage.clear();
    }
  });

  afterEach(() => {
    setActiveWorkspace(null);
    setAvailableWorkspaces([]);
  });

  describe('getActiveWorkspace / setActiveWorkspace', () => {
    it('returns null when no workspace is set', () => {
      expect(getActiveWorkspace()).toBeNull();
    });

    it('returns workspace after setting it', () => {
      setActiveWorkspace('team-a');
      expect(getActiveWorkspace()).toBe('team-a');
    });

    it('persists workspace to localStorage', () => {
      setActiveWorkspace('team-b');
      const stored = window.localStorage.getItem('mlflow.activeWorkspace');
      expect(stored).toBe('team-b');
    });

    it('removes from localStorage when setting to null', () => {
      setActiveWorkspace('team-c');
      expect(window.localStorage.getItem('mlflow.activeWorkspace')).toBe('team-c');

      setActiveWorkspace(null);
      expect(window.localStorage.getItem('mlflow.activeWorkspace')).toBeNull();
    });

    it('notifies listeners when workspace changes', () => {
      const listener = jest.fn();
      subscribeToWorkspaceChanges(listener);

      // Initial call
      expect(listener).toHaveBeenCalledWith(null);

      setActiveWorkspace('team-d');
      expect(listener).toHaveBeenCalledWith('team-d');

      setActiveWorkspace('team-e');
      expect(listener).toHaveBeenCalledWith('team-e');
    });
  });

  describe('extractWorkspaceFromPathname', () => {
    it('returns null for paths without workspace prefix', () => {
      expect(extractWorkspaceFromPathname('/experiments')).toBeNull();
      expect(extractWorkspaceFromPathname('/models/123')).toBeNull();
    });

    it('returns null for empty path', () => {
      expect(extractWorkspaceFromPathname('')).toBeNull();
    });

    it('extracts workspace name from valid workspace path', () => {
      expect(extractWorkspaceFromPathname('/workspaces/default/experiments')).toBe('default');
      expect(extractWorkspaceFromPathname('/workspaces/team-a/models')).toBe('team-a');
    });

    it('rejects URL-encoded workspace names that decode to invalid names', () => {
      // Names with spaces or slashes are not valid workspace names
      expect(extractWorkspaceFromPathname('/workspaces/team%20a/experiments')).toBeNull();
      expect(extractWorkspaceFromPathname('/workspaces/team%2Fb/experiments')).toBeNull();
    });

    it('handles URL-encoded workspace names that are valid', () => {
      // Valid workspace name that happens to be URL-encoded
      expect(extractWorkspaceFromPathname('/workspaces/team-a/experiments')).toBe('team-a');
    });

    it('returns null for malformed workspace paths', () => {
      expect(extractWorkspaceFromPathname('/workspaces/')).toBeNull();
      expect(extractWorkspaceFromPathname('/workspaces')).toBeNull();
    });
  });

  describe('subscribeToWorkspaceChanges', () => {
    it('calls listener immediately with current workspace', () => {
      setActiveWorkspace('initial');
      const listener = jest.fn();

      subscribeToWorkspaceChanges(listener);

      expect(listener).toHaveBeenCalledWith('initial');
    });

    it('calls listener when workspace changes', () => {
      const listener = jest.fn();
      subscribeToWorkspaceChanges(listener);

      listener.mockClear();

      setActiveWorkspace('changed');
      expect(listener).toHaveBeenCalledWith('changed');
    });

    it('returns unsubscribe function that removes listener', () => {
      const listener = jest.fn();
      const unsubscribe = subscribeToWorkspaceChanges(listener);

      listener.mockClear();
      unsubscribe();

      setActiveWorkspace('should-not-notify');
      expect(listener).not.toHaveBeenCalled();
    });
  });

  describe('getActiveWorkspace', () => {
    it('returns the active workspace', () => {
      expect(getActiveWorkspace()).toBeNull();

      setActiveWorkspace('workspace-1');
      expect(getActiveWorkspace()).toBe('workspace-1');
    });
  });

  describe('getAvailableWorkspaces / setAvailableWorkspaces', () => {
    it('gets and sets available workspaces', () => {
      expect(getAvailableWorkspaces()).toEqual([]);

      setAvailableWorkspaces(['default', 'team-a']);
      expect(getAvailableWorkspaces()).toEqual(['default', 'team-a']);
    });
  });

  describe('prefixRouteWithWorkspace', () => {
    beforeEach(() => {
      setActiveWorkspace('default');
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

    it('prefixes absolute paths with workspace', () => {
      expect(prefixRouteWithWorkspace('/experiments')).toBe('/workspaces/default/experiments');
      expect(prefixRouteWithWorkspace('/models/123')).toBe('/workspaces/default/models/123');
    });

    it('handles hash fragments correctly', () => {
      expect(prefixRouteWithWorkspace('#/experiments')).toBe('#/workspaces/default/experiments');
      expect(prefixRouteWithWorkspace('/experiments#section')).toBe('/workspaces/default/experiments#section');
    });

    it('handles query strings correctly', () => {
      expect(prefixRouteWithWorkspace('/experiments?search=test')).toBe('/workspaces/default/experiments?search=test');
      expect(prefixRouteWithWorkspace('/models?filter=active#top')).toBe(
        '/workspaces/default/models?filter=active#top',
      );
    });

    it('does not double-prefix already workspace-prefixed paths', () => {
      const path = '/workspaces/team-a/experiments';
      expect(prefixRouteWithWorkspace(path)).toBe(path);
    });

    it('returns path unprefixed when no workspace set', () => {
      setActiveWorkspace(null);
      expect(prefixRouteWithWorkspace('/experiments')).toBe('/experiments');
    });

    it('handles root path correctly', () => {
      expect(prefixRouteWithWorkspace('/')).toBe('/workspaces/default');
    });

    it('uses different workspace when set', () => {
      setActiveWorkspace('team-a');
      expect(prefixRouteWithWorkspace('/experiments')).toBe('/workspaces/team-a/experiments');
    });

    it('encodes workspace name in URL', () => {
      setActiveWorkspace('team with spaces');
      expect(prefixRouteWithWorkspace('/experiments')).toBe('/workspaces/team%20with%20spaces/experiments');
    });
  });

  describe('prefixPathnameWithWorkspace', () => {
    beforeEach(() => {
      setActiveWorkspace('default');
    });

    it('returns workspace path for undefined pathname', () => {
      expect(prefixPathnameWithWorkspace(undefined)).toBe('/workspaces/default');
    });

    it('returns original for absolute URLs', () => {
      expect(prefixPathnameWithWorkspace('https://example.com')).toBe('https://example.com');
    });

    it('returns original for non-absolute paths', () => {
      expect(prefixPathnameWithWorkspace('relative/path')).toBe('relative/path');
    });

    it('prefixes pathname with workspace', () => {
      expect(prefixPathnameWithWorkspace('/experiments')).toBe('/workspaces/default/experiments');
      expect(prefixPathnameWithWorkspace('/models/123')).toBe('/workspaces/default/models/123');
    });

    it('handles root path correctly', () => {
      expect(prefixPathnameWithWorkspace('/')).toBe('/workspaces/default');
      expect(prefixPathnameWithWorkspace('')).toBe('/workspaces/default');
    });

    it('does not double-prefix workspace paths', () => {
      const path = '/workspaces/team-a/experiments';
      expect(prefixPathnameWithWorkspace(path)).toBe(path);
    });

    it('returns path without workspace when feature disabled', () => {
      getWorkspacesEnabledSyncMock.mockReturnValue(false);
      expect(prefixPathnameWithWorkspace('/experiments')).toBe('/experiments');
    });

    it('uses active workspace when set', () => {
      setActiveWorkspace('team-b');
      expect(prefixPathnameWithWorkspace('/models')).toBe('/workspaces/team-b/models');
    });
  });
});

import { describe, jest, beforeEach, afterEach, it, expect } from '@jest/globals';
import {
  getActiveWorkspace,
  setActiveWorkspace,
  extractWorkspaceFromPathname,
  buildWorkspacePath,
  subscribeToWorkspaceChanges,
  getCurrentWorkspace,
  setAvailableWorkspaces,
  getAvailableWorkspaces,
  hasWorkspaceAccess,
  prefixRouteWithWorkspace,
  prefixPathnameWithWorkspace,
  DEFAULT_WORKSPACE_NAME,
} from './WorkspaceUtils';
import { getWorkspacesEnabledSync } from './ServerFeaturesContext';

jest.mock('./ServerFeaturesContext', () => ({
  ...jest.requireActual<typeof import('./ServerFeaturesContext')>('./ServerFeaturesContext'),
  getWorkspacesEnabledSync: jest.fn(),
}));

const getWorkspacesEnabledSyncMock = jest.mocked(getWorkspacesEnabledSync);

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

    it('handles URL-encoded workspace names', () => {
      expect(extractWorkspaceFromPathname('/workspaces/team%20a/experiments')).toBe('team a');
      expect(extractWorkspaceFromPathname('/workspaces/team%2Fb/experiments')).toBe('team/b');
    });

    it('returns null for malformed workspace paths', () => {
      expect(extractWorkspaceFromPathname('/workspaces/')).toBeNull();
      expect(extractWorkspaceFromPathname('/workspaces')).toBeNull();
    });
  });

  describe('buildWorkspacePath', () => {
    it('builds correct path for simple workspace name', () => {
      expect(buildWorkspacePath('default')).toBe('workspaces/default');
      expect(buildWorkspacePath('team-a')).toBe('workspaces/team-a');
    });

    it('encodes special characters in workspace name', () => {
      expect(buildWorkspacePath('team a')).toBe('workspaces/team%20a');
      expect(buildWorkspacePath('team/b')).toBe('workspaces/team%2Fb');
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

  describe('getCurrentWorkspace', () => {
    it('returns the active workspace', () => {
      expect(getCurrentWorkspace()).toBeNull();

      setActiveWorkspace('workspace-1');
      expect(getCurrentWorkspace()).toBe('workspace-1');
    });
  });

  describe('hasWorkspaceAccess', () => {
    it('returns true when workspace is null', () => {
      setAvailableWorkspaces(['default', 'team-a']);
      expect(hasWorkspaceAccess(null)).toBe(true);
    });

    it('returns true when available workspaces is empty (not loaded)', () => {
      setAvailableWorkspaces([]);
      expect(hasWorkspaceAccess('team-a')).toBe(true);
    });

    it('returns true when workspace is in available list', () => {
      setAvailableWorkspaces(['default', 'team-a', 'team-b']);
      expect(hasWorkspaceAccess('team-a')).toBe(true);
      expect(hasWorkspaceAccess('default')).toBe(true);
    });

    it('returns false when workspace is not in available list', () => {
      setAvailableWorkspaces(['default', 'team-a']);
      expect(hasWorkspaceAccess('team-b')).toBe(false);
      expect(hasWorkspaceAccess('nonexistent')).toBe(false);
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
      expect(prefixRouteWithWorkspace('/experiments?search=test')).toBe(
        '/workspaces/default/experiments?search=test',
      );
      expect(prefixRouteWithWorkspace('/models?filter=active#top')).toBe(
        '/workspaces/default/models?filter=active#top',
      );
    });

    it('does not double-prefix already workspace-prefixed paths', () => {
      const path = '/workspaces/team-a/experiments';
      expect(prefixRouteWithWorkspace(path)).toBe(path);
    });

    it('falls back to DEFAULT_WORKSPACE_NAME when no workspace set', () => {
      setActiveWorkspace(null);
      expect(prefixRouteWithWorkspace('/experiments')).toBe(
        `/workspaces/${DEFAULT_WORKSPACE_NAME}/experiments`,
      );
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

import { describe, it, expect } from '@jest/globals';
import { matchPathWithWorkspace, prefixRoutePathWithWorkspace } from './WorkspaceRouteUtils';

describe('prefixRoutePathWithWorkspace', () => {
  it('returns undefined for empty path', () => {
    expect(prefixRoutePathWithWorkspace(undefined)).toBeUndefined();
  });

  it('prefixes absolute routes', () => {
    expect(prefixRoutePathWithWorkspace('/experiments/:experimentId/runs')).toEqual(
      '/workspaces/:workspaceName/experiments/:experimentId/runs',
    );
  });

  it('prefixes relative routes', () => {
    expect(prefixRoutePathWithWorkspace('experiments/:experimentId/runs')).toEqual(
      '/workspaces/:workspaceName/experiments/:experimentId/runs',
    );
  });

  it('does not double-prefix workspace routes', () => {
    const route = '/workspaces/:workspaceName/experiments/:experimentId/runs';
    expect(prefixRoutePathWithWorkspace(route)).toEqual(route);
  });

  it('normalizes root and wildcard routes', () => {
    expect(prefixRoutePathWithWorkspace('/')).toEqual('/workspaces/:workspaceName');
    expect(prefixRoutePathWithWorkspace('*')).toEqual('/workspaces/:workspaceName/*');
  });
});

describe('matchPathWithWorkspace', () => {
  const routePath = '/experiments/:experimentId/runs';

  it('matches non-workspace paths', () => {
    const match = matchPathWithWorkspace(routePath, '/experiments/7/runs');
    expect(match?.params['experimentId']).toEqual('7');
  });

  it('matches workspace-prefixed paths', () => {
    const match = matchPathWithWorkspace(routePath, '/workspaces/default/experiments/42/runs');
    expect(match?.params['experimentId']).toEqual('42');
  });

  it('returns null when path does not match', () => {
    expect(matchPathWithWorkspace(routePath, '/models/1')).toBeNull();
  });
});

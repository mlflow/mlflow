import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { AdminApi } from './api';
import { useResourceOptionsQuery, useRolesQuery, useUsersQuery, useUserPermissionsQuery } from './hooks';

jest.mock('./api', () => ({
  ...jest.requireActual<typeof import('./api')>('./api'),
  AdminApi: {
    listExperimentsLite: jest.fn(),
    listRegisteredModelsLite: jest.fn(),
    listPromptsLite: jest.fn(),
    listScorersLite: jest.fn(),
    listGatewaySecretsLite: jest.fn(),
    listGatewayEndpointsLite: jest.fn(),
    listUsers: jest.fn(),
    listRoles: jest.fn(),
    listUserPermissions: jest.fn(),
  },
}));

const mockedApi = AdminApi as jest.Mocked<typeof AdminApi>;

const makeWrapper = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
};

beforeEach(() => {
  jest.clearAllMocks();
});

describe('useResourceOptionsQuery — happy paths per resource type', () => {
  it('maps experiments to {id: experiment_id, name}', async () => {
    mockedApi.listExperimentsLite.mockResolvedValueOnce({
      experiments: [
        { experiment_id: '1', name: 'fraud-detection' },
        { experiment_id: '2', name: 'churn' },
      ],
    });
    const { result } = renderHook(() => useResourceOptionsQuery('experiment'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.options).toEqual([
      { id: '1', name: 'fraud-detection' },
      { id: '2', name: 'churn' },
    ]);
  });

  it('maps registered_models with id == name (no separate id field)', async () => {
    mockedApi.listRegisteredModelsLite.mockResolvedValueOnce({
      registered_models: [{ name: 'my-model' }],
    });
    const { result } = renderHook(() => useResourceOptionsQuery('registered_model'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.options).toEqual([{ id: 'my-model', name: 'my-model' }]);
  });

  it('maps prompts off ``registered_models`` (tag-filter search response shape)', async () => {
    mockedApi.listPromptsLite.mockResolvedValueOnce({
      registered_models: [{ name: 'greeting' }, { name: 'farewell' }],
    });
    const { result } = renderHook(() => useResourceOptionsQuery('prompt'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.options).toEqual([
      { id: 'greeting', name: 'greeting' },
      { id: 'farewell', name: 'farewell' },
    ]);
  });

  it('maps scorers to composite id + disambiguated label', async () => {
    // ``id`` is computed client-side via ``scorerResourcePattern`` (mirrors
    // ``SqlAlchemyStore._scorer_pattern``) so the staged grant submits the
    // exact byte sequence the backend persists. The experiment name is joined
    // in client-side from a parallel ``listExperimentsLite`` fetch (kept off
    // the ``Scorer`` proto — UI concern).
    mockedApi.listExperimentsLite.mockResolvedValueOnce({
      experiments: [
        { experiment_id: '1', name: 'fraud' },
        { experiment_id: '2', name: 'churn' },
      ],
    });
    mockedApi.listScorersLite.mockResolvedValueOnce({
      scorers: [
        { experiment_id: 1, scorer_name: 'accuracy' },
        { experiment_id: 2, scorer_name: 'accuracy' },
      ],
    });
    const { result } = renderHook(() => useResourceOptionsQuery('scorer'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.options).toEqual([
      { id: '1/accuracy', name: 'accuracy (in fraud)' },
      { id: '2/accuracy', name: 'accuracy (in churn)' },
    ]);
  });

  it('falls back to experiment_id in the label when the experiment is not in the map', async () => {
    // Defensive: if the parallel ``listExperimentsLite`` doesn't include the
    // scorer's experiment (e.g., the experiment was deleted, or the user
    // can't read it), the picker shouldn't break — fall back to the id.
    mockedApi.listExperimentsLite.mockResolvedValueOnce({ experiments: [] });
    mockedApi.listScorersLite.mockResolvedValueOnce({
      scorers: [{ experiment_id: 7, scorer_name: 'accuracy' }],
    });
    const { result } = renderHook(() => useResourceOptionsQuery('scorer'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.options).toEqual([{ id: '7/accuracy', name: 'accuracy (in 7)' }]);
  });

  it('encodes scorer names like Python urllib quote(safe="")', async () => {
    // ``encodeURIComponent`` leaves ``*'!()`` alone but Python's ``quote``
    // (and the server's ``_scorer_pattern``) percent-encodes them. The picker
    // must match server encoding byte-for-byte so the grant resolves on a
    // subsequent permission check. Includes ``*`` to also pin that a scorer
    // literally named ``*`` doesn't slip through as the wildcard pattern.
    mockedApi.listExperimentsLite.mockResolvedValueOnce({
      experiments: [
        { experiment_id: '1', name: 'fraud' },
        { experiment_id: '2', name: 'churn' },
      ],
    });
    mockedApi.listScorersLite.mockResolvedValueOnce({
      scorers: [
        { experiment_id: 1, scorer_name: '*' },
        { experiment_id: 2, scorer_name: 'hello(world)!' },
      ],
    });
    const { result } = renderHook(() => useResourceOptionsQuery('scorer'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    const ids = result.current.options.map((o) => o.id);
    expect(ids).toEqual(['1/%2A', '2/hello%28world%29%21']);
  });

  it('maps gateway secrets to {id: secret_id, name: secret_name}', async () => {
    mockedApi.listGatewaySecretsLite.mockResolvedValueOnce({
      secrets: [{ secret_id: 's1', secret_name: 'openai-key' }],
    });
    const { result } = renderHook(() => useResourceOptionsQuery('gateway_secret'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.options).toEqual([{ id: 's1', name: 'openai-key' }]);
  });

  it('maps gateway endpoints to {id: endpoint_id, name}', async () => {
    mockedApi.listGatewayEndpointsLite.mockResolvedValueOnce({
      endpoints: [{ endpoint_id: 'e1', name: 'gpt-4' }],
    });
    const { result } = renderHook(() => useResourceOptionsQuery('gateway_endpoint'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.options).toEqual([{ id: 'e1', name: 'gpt-4' }]);
  });
});

describe('useResourceOptionsQuery — scorer picker combines two queries', () => {
  it('stays isLoading until both scorers and experiments resolve', async () => {
    // The scorer picker depends on both endpoints (scorers for ids, experiments
    // for the disambiguating label). Pin that ``isLoading`` reflects ``OR`` of
    // the two — a slow experiments fetch holds the picker until names arrive.
    mockedApi.listScorersLite.mockResolvedValueOnce({
      scorers: [{ experiment_id: 1, scorer_name: 'accuracy' }],
    });
    // Pending forever — picker should stay loading.
    mockedApi.listExperimentsLite.mockReturnValueOnce(new Promise(() => {}));
    const { result } = renderHook(() => useResourceOptionsQuery('scorer'), {
      wrapper: makeWrapper(),
    });
    // Wait for scorers to resolve but experiments still pending.
    await waitFor(() => expect(mockedApi.listScorersLite).toHaveBeenCalled());
    expect(result.current.isLoading).toBe(true);
  });

  it('surfaces a scorers-side error', async () => {
    mockedApi.listExperimentsLite.mockResolvedValueOnce({ experiments: [] });
    const scorerErr = new Error('scorer fetch failed');
    mockedApi.listScorersLite.mockRejectedValueOnce(scorerErr);
    const { result } = renderHook(() => useResourceOptionsQuery('scorer'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.error).toBeTruthy());
    expect(result.current.error).toBe(scorerErr);
  });

  it('surfaces an experiments-side error when scorers succeed', async () => {
    // If the parallel experiments lookup fails, the picker can't render the
    // disambiguating label reliably. Current contract: propagate the error
    // rather than silently rendering ``experiment_id`` labels everywhere.
    const expErr = new Error('experiments fetch failed');
    mockedApi.listScorersLite.mockResolvedValueOnce({
      scorers: [{ experiment_id: 1, scorer_name: 'accuracy' }],
    });
    mockedApi.listExperimentsLite.mockRejectedValueOnce(expErr);
    const { result } = renderHook(() => useResourceOptionsQuery('scorer'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.error).toBeTruthy());
    expect(result.current.error).toBe(expErr);
  });
});

describe('useResourceOptionsQuery — wildcard collision guard', () => {
  it('drops registered_model named "*" so it cannot be picked as a specific grant', async () => {
    // A registered model literally named ``*`` would land as ``resource_id='*'``
    // and silently grant access to ALL registered models — filter it out at
    // the picker layer.
    mockedApi.listRegisteredModelsLite.mockResolvedValueOnce({
      registered_models: [{ name: 'fraud' }, { name: '*' }, { name: 'churn' }],
    });
    const { result } = renderHook(() => useResourceOptionsQuery('registered_model'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.options.map((o) => o.id)).toEqual(['fraud', 'churn']);
  });

  it('drops prompt named "*"', async () => {
    mockedApi.listPromptsLite.mockResolvedValueOnce({
      registered_models: [{ name: 'greeting' }, { name: '*' }],
    });
    const { result } = renderHook(() => useResourceOptionsQuery('prompt'), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.options.map((o) => o.id)).toEqual(['greeting']);
  });
});

const namedRole = {
  id: 7,
  name: 'editors',
  workspace: 'default',
  description: '',
  permissions: [],
};
const syntheticRole = {
  id: 42,
  name: '__user_1__',
  workspace: 'default',
  description: '',
  permissions: [],
};

describe('useUsersQuery', () => {
  it('drops synthetic __user_<id>__ roles from each user', async () => {
    mockedApi.listUsers.mockResolvedValueOnce({
      users: [
        { id: 1, username: 'pat', is_admin: false, roles: [namedRole, syntheticRole] },
        { id: 2, username: 'alex', is_admin: false, roles: [syntheticRole] },
      ],
    });

    const { result } = renderHook(() => useUsersQuery(), { wrapper: makeWrapper() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.users[0].roles).toEqual([namedRole]);
    expect(result.current.data?.users[1].roles).toEqual([]);
  });
});

describe('useRolesQuery', () => {
  it('drops synthetic __user_<id>__ roles from the response', async () => {
    mockedApi.listRoles.mockResolvedValueOnce({ roles: [namedRole, syntheticRole] });

    const { result } = renderHook(() => useRolesQuery(), { wrapper: makeWrapper() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.roles).toEqual([namedRole]);
  });
});

describe('workspace-scoped queries — switching workspace re-fetches with the new header', () => {
  // The same QueryClient is shared across both hooks in each test so cache
  // behavior (key separation, prefix invalidation) matches what the modals see.
  const sharedWrapper = () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };

  it('useResourceOptionsQuery re-fetches experiments when workspace prop changes', async () => {
    // Workspace A returns one set of experiments; switching to B returns a
    // different set. Pins that the cache key includes ``workspace`` (otherwise
    // React Query would serve A's stale entry under the same key).
    mockedApi.listExperimentsLite.mockImplementation((workspace?: string) => {
      if (workspace === 'wsA') {
        return Promise.resolve({
          experiments: [{ experiment_id: '1', name: 'fraud-A' }],
        });
      }
      return Promise.resolve({ experiments: [{ experiment_id: '2', name: 'churn-B' }] });
    });

    const wrapper = sharedWrapper();
    const { result, rerender } = renderHook(
      ({ workspace }: { workspace: string }) => useResourceOptionsQuery('experiment', workspace),
      { initialProps: { workspace: 'wsA' }, wrapper },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.options).toEqual([{ id: '1', name: 'fraud-A' }]);

    rerender({ workspace: 'wsB' });
    await waitFor(() => expect(result.current.options).toEqual([{ id: '2', name: 'churn-B' }]));

    // Two separate fetches: one per workspace key.
    expect(mockedApi.listExperimentsLite).toHaveBeenCalledTimes(2);
    expect(mockedApi.listExperimentsLite).toHaveBeenNthCalledWith(1, 'wsA');
    expect(mockedApi.listExperimentsLite).toHaveBeenNthCalledWith(2, 'wsB');
  });

  it('useResourceOptionsQuery cycles A → B → A correctly (each workspace gets its own data)', async () => {
    // Pin that the cache key keeps per-workspace data separate even after
    // round-trip switching. React Query may issue a background refetch when
    // returning to A (default ``staleTime: 0``), so we don't assert on call
    // count — only on the data the consumer sees being correct each step.
    mockedApi.listExperimentsLite.mockImplementation((workspace?: string) =>
      Promise.resolve({
        experiments: [{ experiment_id: workspace === 'wsA' ? '1' : '2', name: `exp-${workspace}` }],
      }),
    );

    const wrapper = sharedWrapper();
    const { result, rerender } = renderHook(
      ({ workspace }: { workspace: string }) => useResourceOptionsQuery('experiment', workspace),
      { initialProps: { workspace: 'wsA' }, wrapper },
    );
    await waitFor(() => expect(result.current.options[0]?.name).toBe('exp-wsA'));
    rerender({ workspace: 'wsB' });
    await waitFor(() => expect(result.current.options[0]?.name).toBe('exp-wsB'));
    rerender({ workspace: 'wsA' });
    await waitFor(() => expect(result.current.options[0]?.name).toBe('exp-wsA'));
  });

  it('useUserPermissionsQuery re-fetches when workspace changes', async () => {
    // The pre-fill bug we just fixed: ``EditAccessModal`` would fetch the
    // user's grants from the session-active workspace, then revoke against
    // whatever the dropdown selected. Pin that the hook is now workspace-keyed
    // so pre-filled rows match the workspace the revoke will target.
    mockedApi.listUserPermissions.mockImplementation((_username: string, workspace?: string) =>
      Promise.resolve({
        is_admin: false,
        permissions: [
          {
            role_id: 42,
            role_name: '__user_42__',
            workspace: workspace ?? 'default',
            resource_type: 'experiment',
            resource_pattern: workspace === 'wsA' ? 'exp-A' : 'exp-B',
            permission: 'READ',
          },
        ],
      }),
    );

    const wrapper = sharedWrapper();
    const { result, rerender } = renderHook(
      ({ workspace }: { workspace: string }) => useUserPermissionsQuery('pat', workspace),
      { initialProps: { workspace: 'wsA' }, wrapper },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data?.permissions?.[0]?.resource_pattern).toBe('exp-A');

    rerender({ workspace: 'wsB' });
    await waitFor(() => expect(result.current.data?.permissions?.[0]?.resource_pattern).toBe('exp-B'));

    expect(mockedApi.listUserPermissions).toHaveBeenCalledTimes(2);
    expect(mockedApi.listUserPermissions).toHaveBeenNthCalledWith(1, 'pat', 'wsA');
    expect(mockedApi.listUserPermissions).toHaveBeenNthCalledWith(2, 'pat', 'wsB');
  });

  it('different resource types and workspaces share a single QueryClient without cross-contamination', async () => {
    // Pins that the cache identity is (resourceType, workspace), not just
    // resourceType. The modal hosts pickers for several types simultaneously
    // (only one is ``enabled`` at a time, but they all share a key namespace);
    // switching workspace must not bleed type-A's data into type-B's slot.
    mockedApi.listExperimentsLite.mockResolvedValueOnce({
      experiments: [{ experiment_id: '1', name: 'wsA-exp' }],
    });
    mockedApi.listRegisteredModelsLite.mockResolvedValueOnce({
      registered_models: [{ name: 'wsB-model' }],
    });

    const wrapper = sharedWrapper();
    const { result: experimentsInA } = renderHook(() => useResourceOptionsQuery('experiment', 'wsA'), { wrapper });
    const { result: modelsInB } = renderHook(() => useResourceOptionsQuery('registered_model', 'wsB'), { wrapper });

    await waitFor(() => expect(experimentsInA.current.isLoading).toBe(false));
    await waitFor(() => expect(modelsInB.current.isLoading).toBe(false));

    expect(experimentsInA.current.options).toEqual([{ id: '1', name: 'wsA-exp' }]);
    expect(modelsInB.current.options).toEqual([{ id: 'wsB-model', name: 'wsB-model' }]);
    // Each hit its own endpoint once — neither call leaked into the other.
    expect(mockedApi.listExperimentsLite).toHaveBeenCalledTimes(1);
    expect(mockedApi.listRegisteredModelsLite).toHaveBeenCalledTimes(1);
  });
});

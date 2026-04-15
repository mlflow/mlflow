import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, waitFor } from '@testing-library/react';
import React from 'react';

import { WorkspaceRouterSync } from './MlflowRouter';
import { useLocation, useNavigate, useSearchParams } from './common/utils/RoutingUtils';
import { useWorkspaces } from './workspaces/hooks/useWorkspaces';
import {
  extractWorkspaceFromSearchParams,
  getActiveWorkspace,
  getLastUsedWorkspace,
  isGlobalRoute,
  setActiveWorkspace,
  setLastUsedWorkspace,
} from './workspaces/utils/WorkspaceUtils';

jest.mock('./common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('./common/utils/RoutingUtils')>('./common/utils/RoutingUtils'),
  useLocation: jest.fn(),
  useNavigate: jest.fn(),
  useSearchParams: jest.fn(),
}));

jest.mock('./workspaces/hooks/useWorkspaces', () => ({
  useWorkspaces: jest.fn(),
}));

jest.mock('./workspaces/utils/WorkspaceUtils', () => ({
  ...jest.requireActual<typeof import('./workspaces/utils/WorkspaceUtils')>('./workspaces/utils/WorkspaceUtils'),
  extractWorkspaceFromSearchParams: jest.fn(),
  getActiveWorkspace: jest.fn(),
  getLastUsedWorkspace: jest.fn(),
  isGlobalRoute: jest.fn(),
  setActiveWorkspace: jest.fn(),
  setLastUsedWorkspace: jest.fn(),
}));

const useLocationMock = jest.mocked(useLocation);
const useNavigateMock = jest.mocked(useNavigate);
const useSearchParamsMock = jest.mocked(useSearchParams);
const useWorkspacesMock = jest.mocked(useWorkspaces);
const extractWorkspaceFromSearchParamsMock = jest.mocked(extractWorkspaceFromSearchParams);
const getActiveWorkspaceMock = jest.mocked(getActiveWorkspace);
const getLastUsedWorkspaceMock = jest.mocked(getLastUsedWorkspace);
const isGlobalRouteMock = jest.mocked(isGlobalRoute);
const setActiveWorkspaceMock = jest.mocked(setActiveWorkspace);
const setLastUsedWorkspaceMock = jest.mocked(setLastUsedWorkspace);

type MockUseWorkspacesReturn = ReturnType<typeof useWorkspaces>;

describe('WorkspaceRouterSync', () => {
  const mockNavigate = jest.fn();
  let mockWorkspaceState: MockUseWorkspacesReturn;

  beforeEach(() => {
    jest.clearAllMocks();

    useLocationMock.mockReturnValue({
      pathname: '/experiments',
      search: '?workspace=team-a',
      hash: '',
      state: null,
      key: 'default',
    });
    useNavigateMock.mockReturnValue(mockNavigate);
    useSearchParamsMock.mockReturnValue([new URLSearchParams('workspace=team-a'), jest.fn()]);
    mockWorkspaceState = {
      workspaces: [],
      isLoading: false,
      isError: false,
      refetch: jest.fn() as any,
    } as MockUseWorkspacesReturn;
    useWorkspacesMock.mockImplementation(() => mockWorkspaceState);
    extractWorkspaceFromSearchParamsMock.mockReturnValue('team-a');
    getActiveWorkspaceMock.mockReturnValue(null);
    getLastUsedWorkspaceMock.mockReturnValue(null);
    isGlobalRouteMock.mockReturnValue(false);
  });

  it('keeps a valid workspace query active when listWorkspaces does not return it', async () => {
    render(<WorkspaceRouterSync workspacesEnabled />);

    await waitFor(() => {
      expect(setActiveWorkspaceMock).toHaveBeenCalledWith('team-a');
    });

    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it('keeps a valid workspace query active after the optimistic loading state settles', async () => {
    mockWorkspaceState = {
      ...mockWorkspaceState,
      isLoading: true,
    };

    const { rerender } = render(<WorkspaceRouterSync workspacesEnabled />);

    await waitFor(() => {
      expect(setActiveWorkspaceMock).toHaveBeenCalledWith('team-a');
    });

    jest.clearAllMocks();
    mockWorkspaceState = {
      ...mockWorkspaceState,
      isLoading: false,
      isError: false,
      workspaces: [],
    };
    getActiveWorkspaceMock.mockReturnValue('team-a');

    rerender(<WorkspaceRouterSync workspacesEnabled />);

    await waitFor(() => {
      expect(mockNavigate).not.toHaveBeenCalled();
    });
    expect(setActiveWorkspaceMock).not.toHaveBeenCalled();
  });

  it('keeps the normal happy path when the workspace is returned by listWorkspaces', async () => {
    mockWorkspaceState = {
      ...mockWorkspaceState,
      workspaces: [{ name: 'team-a', description: null, default_artifact_root: null }],
    };

    render(<WorkspaceRouterSync workspacesEnabled />);

    await waitFor(() => {
      expect(setActiveWorkspaceMock).toHaveBeenCalledWith('team-a');
    });

    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it('does not redirect away from a valid workspace query when the workspace list request fails', async () => {
    mockWorkspaceState = {
      ...mockWorkspaceState,
      isError: true,
    };

    render(<WorkspaceRouterSync workspacesEnabled />);

    await waitFor(() => {
      expect(setActiveWorkspaceMock).toHaveBeenCalledWith('team-a');
    });

    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it('clears stale workspace state when workspaces are disabled', async () => {
    getActiveWorkspaceMock.mockReturnValue('team-a');

    render(<WorkspaceRouterSync workspacesEnabled={false} />);

    await waitFor(() => {
      expect(setActiveWorkspaceMock).toHaveBeenCalledWith(null);
      expect(setLastUsedWorkspaceMock).toHaveBeenCalledWith(null);
    });
  });
});

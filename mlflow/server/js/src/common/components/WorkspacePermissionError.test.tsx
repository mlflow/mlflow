import { describe, jest, beforeEach, it, expect } from '@jest/globals';
import { screen } from '@testing-library/react';
import React from 'react';

import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { WorkspacePermissionError } from './WorkspacePermissionError';
import { MemoryRouter } from '../utils/RoutingUtils';
import { fetchAPI } from '../utils/FetchUtils';
import { shouldEnableWorkspaces } from '../utils/FeatureUtils';

jest.mock('../utils/FetchUtils', () => ({
  ...jest.requireActual<typeof import('../utils/FetchUtils')>('../utils/FetchUtils'),
  fetchAPI: jest.fn(),
}));

jest.mock('../utils/FeatureUtils', () => ({
  shouldEnableWorkspaces: jest.fn(),
}));

const fetchAPIMock = jest.mocked(fetchAPI);
const shouldEnableWorkspacesMock = jest.mocked(shouldEnableWorkspaces);

describe('WorkspacePermissionError', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Enable workspaces feature for testing
    shouldEnableWorkspacesMock.mockReturnValue(true);

    // Mock workspaces endpoint so WorkspaceSelector doesn't fail
    fetchAPIMock.mockResolvedValue({ workspaces: [{ name: 'default' }, { name: 'team-a' }] });
  });

  const renderWithProviders = (component: React.ReactElement) => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <MemoryRouter>{component}</MemoryRouter>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  it('renders 403 error view with correct message', () => {
    renderWithProviders(<WorkspacePermissionError workspaceName="team-a" />);

    // Check for error image with 403 alt text
    const errorImage = screen.getByRole('img');
    expect(errorImage).toBeInTheDocument();
    expect(errorImage).toHaveAttribute('alt', '403');
  });

  it('displays the workspace name in the error message', () => {
    renderWithProviders(<WorkspacePermissionError workspaceName="restricted-workspace" />);

    // Check that the workspace name appears in the message
    expect(
      screen.getByText(/You don't have access to workspace: restricted-workspace/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/Please select another workspace/i)).toBeInTheDocument();
  });

  it('renders WorkspaceSelector component for switching workspaces', async () => {
    renderWithProviders(<WorkspacePermissionError workspaceName="team-a" />);

    // The WorkspaceSelector should be rendered (it's a combobox)
    // We can't easily test the exact component, but we can check for the combobox
    const combobox = screen.getByRole('combobox');
    expect(combobox).toBeInTheDocument();
  });

  it('displays error for workspace with special characters', () => {
    renderWithProviders(<WorkspacePermissionError workspaceName="team/special-chars!" />);

    expect(
      screen.getByText(/You don't have access to workspace: team\/special-chars!/i),
    ).toBeInTheDocument();
  });
});

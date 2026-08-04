import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import React from 'react';

import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { ErrorCodes } from '../common/constants';
import { ErrorWrapper } from '../common/utils/ErrorWrapper';
import { MemoryRouter } from '../common/utils/RoutingUtils';
import { MlflowService } from '../experiment-tracking/sdk/MlflowService';
import {
  getActiveWorkspace,
  getLastUsedWorkspace,
  setActiveWorkspace,
  setLastUsedWorkspace,
} from '../workspaces/utils/WorkspaceUtils';
import HomePage from './HomePage';

jest.mock('../experiment-tracking/sdk/MlflowService', () => ({
  MlflowService: {
    searchExperiments: jest.fn(),
  },
}));

jest.mock('../experiment-tracking/components/experiment-page/hooks/useExperimentListQuery', () => ({
  useInvalidateExperimentList: () => jest.fn(),
}));

jest.mock('./components/ExperimentsHomeView', () => ({
  __esModule: true,
  default: ({ error }: { error?: Error | null }) => (
    <div data-testid="experiments-home-view">{error ? error.message : 'Experiments Home View'}</div>
  ),
}));

jest.mock('./components/features', () => ({
  FeaturesSection: () => <div data-testid="features-section">Features Section</div>,
}));

jest.mock('./components/LogTracesDrawerLoader', () => ({
  __esModule: true,
  default: () => <div data-testid="log-traces-drawer">Log Traces Drawer</div>,
}));

jest.mock('../telemetry/TelemetryInfoAlert', () => ({
  TelemetryInfoAlert: () => <div data-testid="telemetry-alert">Telemetry Alert</div>,
}));

jest.mock('../experiment-tracking/components/modals/CreateExperimentModal', () => ({
  CreateExperimentModal: () => null,
}));

const searchExperimentsMock = jest.mocked(MlflowService.searchExperiments);

describe('HomePage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    setActiveWorkspace(null);
    setLastUsedWorkspace(null);
  });

  const renderComponent = (initialEntry = '/?workspace=other-name') => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    return renderWithDesignSystem(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[initialEntry]}>
          <HomePage />
        </MemoryRouter>
      </QueryClientProvider>,
    );
  };

  it('renders a workspace-not-found page for homepage workspace misses', async () => {
    searchExperimentsMock.mockRejectedValue(
      new ErrorWrapper(
        {
          error_code: ErrorCodes.RESOURCE_DOES_NOT_EXIST,
          message: "Workspace 'other-name' not found",
        },
        404,
      ),
    );

    renderComponent();

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Page Not Found');
    });

    expect(
      screen.getByText(
        (_, element) => element?.textContent === 'Workspace "other-name" was not found, go back to the home page.',
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole('link')).toHaveAttribute('href', '/');
  });

  it('clears the remembered workspace when the homepage detects a missing workspace', async () => {
    setActiveWorkspace('other-name');
    setLastUsedWorkspace('other-name');
    searchExperimentsMock.mockRejectedValue(
      new ErrorWrapper(
        {
          error_code: ErrorCodes.RESOURCE_DOES_NOT_EXIST,
          message: "Workspace 'other-name' not found",
        },
        404,
      ),
    );

    renderComponent();

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Page Not Found');
    });

    expect(getActiveWorkspace()).toBeNull();
    expect(getLastUsedWorkspace()).toBeNull();
  });

  it('accepts a looser workspace-not-found backend message', async () => {
    searchExperimentsMock.mockRejectedValue(
      new ErrorWrapper(
        {
          error_code: ErrorCodes.RESOURCE_DOES_NOT_EXIST,
          message: "Workspace 'other-name' not found in workspace store",
        },
        404,
      ),
    );

    renderComponent();

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Page Not Found');
    });
  });

  it('keeps generic experiment-load errors in the normal homepage view', async () => {
    searchExperimentsMock.mockRejectedValue(
      new ErrorWrapper(
        {
          error_code: ErrorCodes.INTERNAL_ERROR,
          message: 'Boom',
        },
        500,
      ),
    );

    renderComponent();

    await waitFor(() => {
      expect(screen.getByTestId('experiments-home-view')).toBeInTheDocument();
    });

    expect(
      screen.queryByText(
        (_, element) => element?.textContent === 'Workspace "other-name" was not found, go back to the home page.',
      ),
    ).not.toBeInTheDocument();
  });

  it('does not treat unrelated resource-missing errors as workspace-not-found', async () => {
    searchExperimentsMock.mockRejectedValue(
      new ErrorWrapper(
        {
          error_code: ErrorCodes.RESOURCE_DOES_NOT_EXIST,
          message: 'Experiment not found',
        },
        404,
      ),
    );

    renderComponent();

    await waitFor(() => {
      expect(screen.getByTestId('experiments-home-view')).toBeInTheDocument();
    });

    expect(screen.queryByRole('heading', { level: 1, name: 'Page Not Found' })).not.toBeInTheDocument();
  });
});

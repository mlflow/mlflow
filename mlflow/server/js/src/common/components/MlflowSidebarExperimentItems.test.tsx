import { describe, jest, test, expect, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { MlflowSidebarExperimentItems } from './MlflowSidebarExperimentItems';
import { MemoryRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../common/utils/reactQueryHooks';
import { MockedReduxStoreProvider } from '../../common/utils/TestUtils';
import { WorkflowType } from '../contexts/WorkflowTypeContext';
import { ExperimentPageTabName } from '../../experiment-tracking/constants';

jest.mock('../../experiment-tracking/components/experiment-page/hooks/useExperimentEvaluationRunsData', () => ({
  useExperimentEvaluationRunsData: jest.fn(() => ({ trainingRuns: [] })),
}));

jest.mock('../../experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(() => ({ data: { name: 'Test Experiment' }, loading: false })),
}));

jest.mock('@mlflow/mlflow/src/telemetry/hooks/useLogTelemetryEvent', () => ({
  useLogTelemetryEvent: jest.fn(() => jest.fn()),
}));

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableExperimentOverviewTab: jest.fn(() => true),
}));

const mockUseGetExperimentPageActiveTabByRoute = jest.fn();
jest.mock('../../experiment-tracking/components/experiment-page/hooks/useGetExperimentPageActiveTabByRoute', () => ({
  useGetExperimentPageActiveTabByRoute: () => mockUseGetExperimentPageActiveTabByRoute(),
}));

describe('MlflowSidebarExperimentItems', () => {
  const renderTestComponent = (initialEntries: string[] = ['/experiments/test-123/traces']) => {
    const queryClient = new QueryClient();
    return render(
      <MockedReduxStoreProvider state={{ entities: { experimentTagsByExperimentId: {}, experimentsById: {} } }}>
        <IntlProvider locale="en">
          <QueryClientProvider client={queryClient}>
            <DesignSystemProvider>
              <MemoryRouter initialEntries={initialEntries}>
                <MlflowSidebarExperimentItems
                  collapsed={false}
                  experimentId="test-123"
                  workflowType={WorkflowType.GENAI}
                />
              </MemoryRouter>
            </DesignSystemProvider>
          </QueryClientProvider>
        </IntlProvider>
      </MockedReduxStoreProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseGetExperimentPageActiveTabByRoute.mockReturnValue({ tabName: ExperimentPageTabName.Traces });
  });

  test('preserves query params when navigating between traces-related tabs', () => {
    mockUseGetExperimentPageActiveTabByRoute.mockReturnValue({ tabName: ExperimentPageTabName.Traces });

    renderTestComponent(['/experiments/test-123/traces?startTimeLabel=LAST_24_HOURS&startTime=2024-01-01']);

    // Find the Sessions link (traces-related tab)
    const sessionsLink = screen.getByText('Sessions').closest('a');
    expect(sessionsLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
    expect(sessionsLink).toHaveAttribute('href', expect.stringContaining('startTime=2024-01-01'));

    // Find the Overview link (traces-related tab)
    const overviewLink = screen.getByText('Overview').closest('a');
    expect(overviewLink).toHaveAttribute('href', expect.stringContaining('startTimeLabel=LAST_24_HOURS'));
  });

  test('does not preserve query params when navigating to non-traces-related tabs', () => {
    mockUseGetExperimentPageActiveTabByRoute.mockReturnValue({ tabName: ExperimentPageTabName.Traces });

    renderTestComponent(['/experiments/test-123/traces?startTimeLabel=LAST_24_HOURS']);

    // Find the Datasets link (non-traces-related tab)
    const datasetsLink = screen.getByText('Datasets').closest('a');
    expect(datasetsLink).not.toHaveAttribute('href', expect.stringContaining('startTimeLabel'));
  });

  test('does not preserve query params when navigating from non-traces-related tab', () => {
    mockUseGetExperimentPageActiveTabByRoute.mockReturnValue({ tabName: ExperimentPageTabName.Datasets });

    renderTestComponent(['/experiments/test-123/datasets?startTimeLabel=LAST_24_HOURS']);

    // Even though we have query params, they should not be preserved when coming from Datasets
    const tracesLink = screen.getByText('Traces').closest('a');
    expect(tracesLink).not.toHaveAttribute('href', expect.stringContaining('startTimeLabel'));
  });
});

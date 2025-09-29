import React from 'react';
import { render, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { TabSelectorBar } from './TabSelectorBar';
import { MemoryRouter, useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useExperimentPageViewMode } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/hooks/useExperimentPageViewMode';
import type { ExperimentViewRunsCompareMode } from '@mlflow/mlflow/src/experiment-tracking/types';
import { IntlProvider } from 'react-intl';
import { shouldEnablePromptsTabOnDBPlatform } from '../../../../../../common/utils/FeatureUtils';

import { ExperimentKind } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

const queryClient = new QueryClient();

// Mock the hooks
jest.mock('@mlflow/mlflow/src/common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/RoutingUtils')>(
    '@mlflow/mlflow/src/common/utils/RoutingUtils',
  ),
  useParams: jest.fn(),
}));

jest.mock('@mlflow/mlflow/src/experiment-tracking/components/experiment-page/hooks/useExperimentPageViewMode', () => ({
  useExperimentPageViewMode: jest.fn(),
}));

jest.mock(
  '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/hooks/useExperimentEvaluationRunsData',
  () => ({
    useExperimentEvaluationRunsData: jest.fn().mockReturnValue({
      data: [],
      trainingRuns: [],
      isLoading: false,
      isFetching: false,
      error: null,
      refetch: jest.fn(),
    }),
  }),
);

jest.mock('../../../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../../../common/utils/FeatureUtils')>(
    '../../../../../../common/utils/FeatureUtils',
  ),
  shouldEnablePromptsTabOnDBPlatform: jest.fn(),
}));

const mockShouldEnablePromptsTabOnDBPlatform = jest.mocked(shouldEnablePromptsTabOnDBPlatform);

describe('TabSelectorBar', () => {
  const mockUseParams = jest.mocked(useParams);
  const mockUseExperimentPageViewMode = jest.mocked(useExperimentPageViewMode);

  const renderComponent = (props = {}) => {
    return render(<TabSelectorBar {...props} />, {
      wrapper: ({ children }) => (
        <DesignSystemProvider>
          <QueryClientProvider client={queryClient}>
            <MemoryRouter>
              <IntlProvider locale="en">{children}</IntlProvider>
            </MemoryRouter>
          </QueryClientProvider>
        </DesignSystemProvider>
      ),
    });
  };

  beforeEach(() => {
    jest.clearAllMocks();

    mockShouldEnablePromptsTabOnDBPlatform.mockReturnValue(false);
  });

  test('renders with default props without exploding', () => {
    mockUseParams.mockReturnValue({
      experimentId: '123',
      tabName: 'models',
    });

    mockUseExperimentPageViewMode.mockReturnValue(['MODELS' as ExperimentViewRunsCompareMode, jest.fn()]);

    renderComponent();

    expect(screen.getByTestId('tab-selector-button-text-models-active')).toBeInTheDocument();
  });

  test('highlights active tab correctly', () => {
    // mock being on the models page
    mockUseParams.mockReturnValue({
      experimentId: '123',
      tabName: 'models',
    });
    mockUseExperimentPageViewMode.mockReturnValue(['MODELS' as ExperimentViewRunsCompareMode, jest.fn()]);
    const { rerender } = renderComponent();
    const activeButton1 = screen.getByTestId('tab-selector-button-text-models-active');
    expect(activeButton1).toBeInTheDocument();

    // mock being on the runs page
    mockUseParams.mockReturnValue({
      experimentId: '123',
      tabName: 'runs',
    });

    rerender(<TabSelectorBar />);
    const activeButton2 = screen.getByTestId('tab-selector-button-text-runs-active');
    expect(activeButton2).toBeInTheDocument();
  });
});

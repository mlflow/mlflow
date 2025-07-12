import React from 'react';
import { render, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { TabSelectorBar } from './TabSelectorBar';
import { MemoryRouter, useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useExperimentPageViewMode } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/hooks/useExperimentPageViewMode';
import { ExperimentViewRunsCompareMode } from '@mlflow/mlflow/src/experiment-tracking/types';
import { IntlProvider } from 'react-intl';
import { shouldEnablePromptsTabOnDBPlatform } from '../../../../../../common/utils/FeatureUtils';

import { ExperimentKind } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { DesignSystemProvider } from '@databricks/design-system';

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

    render(<TabSelectorBar />, {
      wrapper: ({ children }) => (
        <DesignSystemProvider>
          <MemoryRouter>
            <IntlProvider locale="en">{children}</IntlProvider>
          </MemoryRouter>
        </DesignSystemProvider>
      ),
    });

    expect(screen.getByTestId('tab-selector-button-text-models-active')).toBeInTheDocument();
  });

  test('highlights active tab correctly', () => {
    // mock being on the models page
    mockUseParams.mockReturnValue({
      experimentId: '123',
      tabName: 'models',
    });
    mockUseExperimentPageViewMode.mockReturnValue(['MODELS' as ExperimentViewRunsCompareMode, jest.fn()]);
    const { rerender } = render(<TabSelectorBar />, {
      wrapper: ({ children }) => (
        <DesignSystemProvider>
          <MemoryRouter>
            <IntlProvider locale="en">{children}</IntlProvider>
          </MemoryRouter>
        </DesignSystemProvider>
      ),
    });
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

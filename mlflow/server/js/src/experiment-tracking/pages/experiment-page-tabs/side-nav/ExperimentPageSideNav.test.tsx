import { describe, jest, test, expect, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { ExperimentPageSideNav } from './ExperimentPageSideNav';
import { ExperimentKind, ExperimentPageTabName } from '../../../constants';
import { MemoryRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../../../common/utils/reactQueryHooks';
import { MockedReduxStoreProvider } from '../../../../common/utils/TestUtils';
jest.mock('../../../components/experiment-page/hooks/useExperimentEvaluationRunsData', () => ({
  useExperimentEvaluationRunsData: jest.fn(() => ({ trainingRuns: [] })),
}));

// Mock useParams from the routing utilss
jest.mock('../../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/RoutingUtils')>(
    '../../../../common/utils/RoutingUtils',
  ),
  useParams: () => ({ experimentId: 'test-experiment-123' }),
}));

jest.mock('@mlflow/mlflow/src/telemetry/hooks/useLogTelemetryEvent', () => ({
  useLogTelemetryEvent: jest.fn(() => jest.fn()),
}));

describe('ExperimentPageSideNav', () => {
  const renderTestComponent = (experimentKind: ExperimentKind, activeTab: ExperimentPageTabName) => {
    const queryClient = new QueryClient();
    return render(
      <MockedReduxStoreProvider state={{ entities: { experimentTagsByExperimentId: {}, experimentsById: {} } }}>
        <IntlProvider locale="en">
          <QueryClientProvider client={queryClient}>
            <DesignSystemProvider>
              <MemoryRouter>
                <ExperimentPageSideNav experimentKind={experimentKind} activeTab={activeTab} />
              </MemoryRouter>
            </DesignSystemProvider>
          </QueryClientProvider>
        </IntlProvider>
      </MockedReduxStoreProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test.each([ExperimentKind.GENAI_DEVELOPMENT, ExperimentKind.GENAI_DEVELOPMENT_INFERRED])(
    'should render GenAI experiment tabs',
    (experimentKind) => {
      renderTestComponent(experimentKind, ExperimentPageTabName.Traces);

      // Check observability section
      expect(screen.getByText('Observability')).toBeInTheDocument();
      expect(screen.getByText('Traces')).toBeInTheDocument();
      expect(screen.getByText('Sessions')).toBeInTheDocument();

      // Check evaluation section
      expect(screen.getByText('Evaluation')).toBeInTheDocument();
      expect(screen.getByText('Datasets')).toBeInTheDocument();

      // Check prompts & versions section
      const versionsSectionHeader = 'Prompts & versions';
      expect(screen.getByText(versionsSectionHeader)).toBeInTheDocument();
      expect(screen.getByText('Agent versions')).toBeInTheDocument();
    },
  );

  test('should not render chat sessions for non-genai', () => {
    renderTestComponent(ExperimentKind.CUSTOM_MODEL_DEVELOPMENT, ExperimentPageTabName.Runs);
    expect(screen.queryByText('Sessions')).not.toBeInTheDocument();
  });

  test.each([
    ExperimentKind.CUSTOM_MODEL_DEVELOPMENT,
    ExperimentKind.FINETUNING,
    ExperimentKind.AUTOML,
    ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED,
  ])('should render custom model development tabs', (experimentKind) => {
    renderTestComponent(experimentKind, ExperimentPageTabName.Runs);

    // Check top-level section tabs
    expect(screen.getByText('Runs')).toBeInTheDocument();
    expect(screen.getByText('Models')).toBeInTheDocument();
    expect(screen.getByText('Traces')).toBeInTheDocument();
  });
});

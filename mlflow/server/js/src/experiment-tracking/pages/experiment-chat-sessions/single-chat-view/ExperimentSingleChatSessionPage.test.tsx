import React from 'react';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { describe, expect, jest, test, beforeEach } from '@jest/globals';
import { shouldEnableModelTraceExplorerCustomTraceView } from '@databricks/web-shared/model-trace-explorer';
import { CustomViewDefinitionProvider } from '@databricks/web-shared/model-trace-explorer/custom-view/CustomViewDefinitionContext';

import ExperimentSingleChatSessionPage from './ExperimentSingleChatSessionPage';

const mockExperimentCustomViewProvider = jest.fn(
  ({ children, experimentId }: { children: React.ReactNode; experimentId?: string }) => (
    <div data-testid="custom-view-provider" data-experiment-id={experimentId}>
      {children}
    </div>
  ),
);

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  CUSTOM_METADATA_COLUMN_ID: 'metadata',
  FilterOperator: {
    CONTAINS: 'CONTAINS',
    EQUALS: 'EQUALS',
  },
  createTraceLocationForExperiment: jest.fn((experimentId: string) => ({ experimentId })),
  createTraceLocationForDestinationPath: jest.fn(),
  doesTraceSupportV4API: jest.fn(() => false),
  useGetTraces: jest.fn(() => ({
    data: [],
    isLoading: false,
    invalidateSingleTraceQuery: jest.fn(),
  })),
  useSearchMlflowTraces: jest.fn(() => ({
    data: [],
    isLoading: false,
  })),
}));

jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  getModelTraceId: jest.fn(
    (trace: { info?: { trace_id?: string }; trace_id?: string }) => trace?.info?.trace_id ?? trace?.trace_id,
  ),
  isEvaluatingTracesInDetailsViewEnabled: jest.fn(() => false),
  isV3ModelTraceInfo: jest.fn(() => true),
  ModelTraceExplorer: jest.fn(() => null),
  ModelTraceExplorerContextProvider: ({ children }: { children: React.ReactNode }) => children,
  ModelTraceExplorerDrawer: ({ children }: { children: React.ReactNode }) => children,
  ModelTraceExplorerRunJudgesContextProvider: ({ children }: { children: React.ReactNode }) => children,
  ModelTraceExplorerUpdateTraceContextProvider: ({ children }: { children: React.ReactNode }) => children,
  ModelTraceExplorerPreferencesProvider: ({ children }: { children: React.ReactNode }) => children,
  shouldEnableAssessmentsInSessions: jest.fn(() => false),
  shouldEnableModelTraceExplorerCustomTraceView: jest.fn(),
  shouldUseTracesV4API: jest.fn(() => false),
}));

jest.mock('../../../hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(() => ({ loading: false })),
}));

jest.mock('@mlflow/mlflow/src/common/utils/RoutingUtils', () => ({
  useLocation: () => ({ search: '' }),
  useParams: () => ({ experimentId: 'experiment-123', sessionId: 'session-456' }),
}));

jest.mock('@mlflow/mlflow/src/assistant', () => ({
  useRegisterAssistantContext: jest.fn(),
}));

jest.mock('../../../components/experiment-page/components/traces-v3/TracesV3Toolbar', () => ({
  TracesV3Toolbar: () => <div data-testid="traces-toolbar" />,
}));

jest.mock('./ExperimentSingleChatSessionSidebar', () => ({
  ExperimentSingleChatSessionSidebar: () => <div data-testid="chat-sidebar" />,
  ExperimentSingleChatSessionSidebarSkeleton: () => <div data-testid="chat-sidebar-skeleton" />,
}));

jest.mock('./ExperimentSingleChatConversation', () => ({
  ExperimentSingleChatConversation: () => <div data-testid="chat-conversation" />,
  ExperimentSingleChatConversationSkeleton: () => <div data-testid="chat-conversation-skeleton" />,
}));

jest.mock('./ExperimentSingleChatSessionMetrics', () => ({
  ExperimentSingleChatSessionMetrics: () => null,
}));

jest.mock('./ExperimentSingleChatSessionScoreResults', () => ({
  ExperimentSingleChatSessionScoreResults: () => null,
}));

jest.mock('../../experiment-evaluation-datasets/components/ExportTracesToDatasetModal', () => ({
  ExportTracesToDatasetModal: () => null,
}));

jest.mock('../../experiment-scorers/hooks/useRunScorerInTracesViewConfiguration', () => ({
  useRunScorerInTracesViewConfiguration: jest.fn(() => ({})),
}));

jest.mock('./useExperimentSingleChatMetrics', () => ({
  useExperimentSingleChatMetrics: jest.fn(() => ({})),
}));

jest.mock('@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils', () => ({
  getTrace: jest.fn(),
}));

jest.mock('../../../components/experiment-page/components/traces-v3/ExperimentCustomViewProvider', () => ({
  ExperimentCustomViewProvider: (props: { children: React.ReactNode; experimentId?: string }) =>
    mockExperimentCustomViewProvider(props),
}));

const renderPage = (withCustomViewProvider = false) =>
  renderWithIntl(
    <DesignSystemProvider>
      {withCustomViewProvider ? (
        <CustomViewDefinitionProvider views={[]} isLoaded>
          <ExperimentSingleChatSessionPage />
        </CustomViewDefinitionProvider>
      ) : (
        <ExperimentSingleChatSessionPage />
      )}
    </DesignSystemProvider>,
  );

describe('ExperimentSingleChatSessionPage', () => {
  beforeEach(() => {
    mockExperimentCustomViewProvider.mockClear();
  });

  test('mounts ExperimentCustomViewProvider with the current experiment ID when custom trace view is enabled', async () => {
    jest.mocked(shouldEnableModelTraceExplorerCustomTraceView).mockReturnValue(true);

    const { getByTestId } = renderPage();

    await waitFor(() => {
      expect(mockExperimentCustomViewProvider).toHaveBeenCalledWith(
        expect.objectContaining({ experimentId: 'experiment-123' }),
      );
    });
    expect(getByTestId('custom-view-provider')).toHaveAttribute('data-experiment-id', 'experiment-123');
  });

  test('omits ExperimentCustomViewProvider when custom trace view is disabled', async () => {
    jest.mocked(shouldEnableModelTraceExplorerCustomTraceView).mockReturnValue(false);

    const { queryByTestId } = renderPage();

    await waitFor(() => {
      expect(mockExperimentCustomViewProvider).not.toHaveBeenCalled();
      expect(queryByTestId('custom-view-provider')).not.toBeInTheDocument();
    });
  });

  test('reuses an outer custom view definition instead of mounting a nested provider', async () => {
    jest.mocked(shouldEnableModelTraceExplorerCustomTraceView).mockReturnValue(true);

    const { queryByTestId } = renderPage(true);

    await waitFor(() => {
      expect(mockExperimentCustomViewProvider).not.toHaveBeenCalled();
      expect(queryByTestId('custom-view-provider')).not.toBeInTheDocument();
    });
  });
});

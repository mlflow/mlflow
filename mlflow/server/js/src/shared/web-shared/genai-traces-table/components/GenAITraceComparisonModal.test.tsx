import { jest, describe, it, expect, afterEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { IntlProvider } from '@databricks/i18n';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { type ModelTrace } from '../../model-trace-explorer';
import { MOCK_TRACE } from '../../model-trace-explorer/ModelTraceExplorer.test-utils';
import { GenAITraceComparisonModal } from './GenAITraceComparisonModal';

jest.mock('../../model-trace-explorer/CompareModelTraceExplorer', () => ({
  CompareModelTraceExplorer: jest.fn(() => <div data-testid="compare-model-trace-explorer" />),
}));

jest.mock('../../model-trace-explorer/ModelTraceExplorerSkeleton', () => ({
  ModelTraceExplorerSkeleton: jest.fn(() => <div data-testid="model-trace-explorer-skeleton" />),
}));

jest.mock('@databricks/web-shared/model-trace-explorer/hooks/useGetTracesById', () => ({
  useGetTracesById: jest.fn().mockReturnValue({ data: null, isLoading: true }),
}));

const getCompareModelTraceExplorerMock = () =>
  (jest.requireMock('../../model-trace-explorer/CompareModelTraceExplorer') as any)
    .CompareModelTraceExplorer as jest.Mock;

const getUseGetTracesByIdMock = () =>
  (jest.requireMock('@databricks/web-shared/model-trace-explorer/hooks/useGetTracesById') as any)
    .useGetTracesById as jest.Mock;

const renderWithProviders = (ui: React.ReactNode) =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <IntlProvider locale="en">
        <DesignSystemProvider>{ui}</DesignSystemProvider>
      </IntlProvider>
    </QueryClientProvider>,
  );

describe('GenAITraceComparisonModal', () => {
  const mockTraceIds = ['trace-1', 'trace-2'];

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('shows skeleton when traces are still being resolved', () => {
    getUseGetTracesByIdMock().mockReturnValue({
      data: undefined,
      isLoading: true,
    });

    renderWithProviders(<GenAITraceComparisonModal traceIds={mockTraceIds} onClose={jest.fn()} />);

    expect(screen.getByTestId('model-trace-explorer-skeleton')).toBeInTheDocument();
    expect(getCompareModelTraceExplorerMock()).not.toHaveBeenCalled();
  });

  it('requests the traces and renders the compare explorer', async () => {
    const resolvedTrace1: ModelTrace = {
      ...MOCK_TRACE,
      info: {
        ...(MOCK_TRACE.info as any),
        trace_id: 'trace-1',
      },
    } as ModelTrace;

    const resolvedTrace2: ModelTrace = {
      ...MOCK_TRACE,
      info: {
        ...(MOCK_TRACE.info as any),
        trace_id: 'trace-2',
      },
    } as ModelTrace;

    const mockUseGetTracesById = getUseGetTracesByIdMock().mockReturnValue({
      data: [resolvedTrace1, resolvedTrace2],
      isLoading: false,
    });

    renderWithProviders(<GenAITraceComparisonModal traceIds={mockTraceIds} onClose={jest.fn()} />);

    expect(mockUseGetTracesById).toHaveBeenCalledWith(mockTraceIds, undefined);
    expect(screen.getAllByTestId('compare-model-trace-explorer')).toHaveLength(1);
    expect(getCompareModelTraceExplorerMock()).toHaveBeenCalledWith(
      expect.objectContaining({
        modelTraces: [resolvedTrace1, resolvedTrace2],
      }),
    );
  });
});

import { jest, describe, it, expect, afterEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { IntlProvider } from '@databricks/i18n';
import { DesignSystemProvider } from '@databricks/design-system';
import type { ModelTrace } from '../../model-trace-explorer';
import { MOCK_TRACE } from '../../model-trace-explorer/ModelTraceExplorer.test-utils';
import { GenAITraceComparisonModal } from './GenAITraceComparisonModal';

jest.mock('../../model-trace-explorer', () => ({
  ModelTraceExplorer: jest.fn(() => <div data-testid="model-trace-explorer" />),
}));

jest.mock('@mlflow/mlflow/src/experiment-tracking/pages/experiment-evaluation-datasets/hooks/useFetchTraces', () => ({
  useFetchTraces: jest.fn(),
}));

const getModelTraceExplorerMock = () =>
  (jest.requireMock('../../model-trace-explorer') as any).ModelTraceExplorer as jest.Mock;
const getUseFetchTracesMock = () =>
  (
    jest.requireMock(
      '@mlflow/mlflow/src/experiment-tracking/pages/experiment-evaluation-datasets/hooks/useFetchTraces',
    ) as any
  ).useFetchTraces as jest.Mock;

const renderWithProviders = (ui: React.ReactNode) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>{ui}</DesignSystemProvider>
    </IntlProvider>,
  );

describe('GenAITraceComparisonModal', () => {
  const mockTraceIds = ['trace-1', 'trace-2'];

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('shows a loading message when traces are still being resolved', () => {
    getUseFetchTracesMock().mockReturnValue({
      data: undefined,
      isLoading: true,
    });

    renderWithProviders(<GenAITraceComparisonModal traceIds={mockTraceIds} onClose={jest.fn()} />);

    expect(screen.getByText('Loading traces…')).toBeInTheDocument();
    expect(getModelTraceExplorerMock()).not.toHaveBeenCalled();
  });

  it('requests the traces and renders the explorer for each resolved trace', async () => {
    const resolvedTrace: ModelTrace = {
      ...MOCK_TRACE,
      info: {
        ...(MOCK_TRACE.info as any),
        trace_id: 'trace-1',
      },
    } as ModelTrace;

    getUseFetchTracesMock().mockReturnValue({
      data: [resolvedTrace],
      isLoading: false,
    });

    renderWithProviders(<GenAITraceComparisonModal traceIds={mockTraceIds} onClose={jest.fn()} />);

    await waitFor(() => expect(getUseFetchTracesMock()).toHaveBeenCalledWith({ traceIds: mockTraceIds }));

    await waitFor(() => expect(screen.getAllByTestId('model-trace-explorer')).toHaveLength(1));

    expect(getModelTraceExplorerMock()).toHaveBeenCalledWith(
      expect.objectContaining({
        modelTrace: resolvedTrace,
        initialActiveView: 'summary',
        isInComparisonView: true,
      }),
      {},
    );
  });
});

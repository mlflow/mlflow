import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { IntlProvider } from '@databricks/i18n';
import { DesignSystemProvider } from '@databricks/design-system';
import type { ModelTrace } from '../../model-trace-explorer';
import { MOCK_TRACE } from '../../model-trace-explorer/ModelTraceExplorer.test-utils';
import type { RunEvaluationTracesDataEntry } from '../types';
import { TraceComparisonModal } from './GenAITraceComparisonModal';

jest.mock('../../model-trace-explorer', () => ({
  ModelTraceExplorer: jest.fn(() => <div data-testid="model-trace-explorer" />),
}));

const getModelTraceExplorerMock = () => jest.requireMock('../../model-trace-explorer').ModelTraceExplorer as jest.Mock;

const renderWithProviders = (ui: React.ReactNode) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>{ui}</DesignSystemProvider>
    </IntlProvider>,
  );

describe('TraceComparisonModal', () => {
  const mockTraces: RunEvaluationTracesDataEntry[] = [
    {
      evaluationId: 'eval-1',
      requestId: 'req-1',
      inputs: {},
      inputsId: 'inputs-1',
      outputs: {},
      targets: {},
      overallAssessments: [],
      responseAssessmentsByName: {},
      metrics: {},
      traceInfo: { trace_id: 'trace-1' } as any,
    },
    {
      evaluationId: 'eval-2',
      requestId: 'req-2',
      inputs: {},
      inputsId: 'inputs-2',
      outputs: {},
      targets: {},
      overallAssessments: [],
      responseAssessmentsByName: {},
      metrics: {},
      traceInfo: { trace_id: 'trace-2' } as any,
    },
  ];

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('shows a loading message when traces are still being resolved', () => {
    renderWithProviders(<TraceComparisonModal traces={mockTraces} onClose={jest.fn()} />);

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
    const getTrace = jest
      .fn<Promise<ModelTrace | undefined>, [string | undefined]>()
      .mockResolvedValueOnce(resolvedTrace)
      .mockResolvedValueOnce(undefined);

    renderWithProviders(<TraceComparisonModal traces={mockTraces} onClose={jest.fn()} getTrace={getTrace} />);

    await waitFor(() => expect(getTrace).toHaveBeenCalledTimes(2));

    expect(getTrace).toHaveBeenNthCalledWith(1, 'trace-1');
    expect(getTrace).toHaveBeenNthCalledWith(2, 'trace-2');

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

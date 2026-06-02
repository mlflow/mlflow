import { describe, beforeAll, beforeEach, afterEach, afterAll, jest, it, expect } from '@jest/globals';
import React from 'react';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AggregationType } from '@databricks/web-shared/model-trace-explorer';

import { RunEvaluationButton } from './RunEvaluationButton';

const mockUseTraceMetricsQuery = jest.fn();
jest.mock('../experiment-overview/hooks/useTraceMetricsQuery', () => ({
  __esModule: true,
  useTraceMetricsQuery: (...args: unknown[]) => mockUseTraceMetricsQuery(...args),
}));

const COPY_BUTTON_SELECTOR = '[data-component-id="mlflow.eval-runs.start-run-modal.copy-snippet"]';

describe('RunEvaluationButton', () => {
  let originalClipboard: typeof navigator.clipboard;

  beforeAll(() => {
    originalClipboard = navigator.clipboard;
  });

  beforeEach(() => {
    Object.defineProperty(global.navigator, 'clipboard', {
      value: { writeText: jest.fn(() => Promise.resolve()) },
      writable: true,
    });
    mockUseTraceMetricsQuery.mockReset();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  afterAll(() => {
    Object.defineProperty(global.navigator, 'clipboard', {
      value: originalClipboard,
      writable: true,
    });
  });

  const renderButton = (experimentId = 'exp-1') =>
    renderWithIntl(
      <DesignSystemProvider>
        <RunEvaluationButton experimentId={experimentId} />
      </DesignSystemProvider>,
    );

  // Open the modal and click its copy button. Returns the string passed to clipboard.writeText.
  const openModalAndCopy = async (): Promise<string> => {
    await userEvent.click(screen.getByRole('button', { name: 'Run evaluation' }));

    const copyButton = await waitFor(() => {
      const el = document.querySelector<HTMLElement>(COPY_BUTTON_SELECTOR);
      expect(el).not.toBeNull();
      return el!;
    });
    await userEvent.click(copyButton);

    expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
    return (navigator.clipboard.writeText as jest.Mock).mock.calls[0][0] as string;
  };

  it('renders the dataset-based snippet and copies it when the experiment has no traces', async () => {
    mockUseTraceMetricsQuery.mockReturnValue({
      data: { data_points: [{ values: { [AggregationType.COUNT]: 0 } }] },
    });

    renderButton('exp-1');

    const copied = await openModalAndCopy();

    expect(copied).toContain('mlflow.set_experiment(experiment_id="exp-1")');
    expect(copied).toContain('eval_dataset = [{');
    expect(copied).toContain('predict_fn=predict');
    expect(copied).not.toContain('mlflow.search_traces');
  });

  it('renders the trace-based snippet and copies it when the experiment has traces', async () => {
    mockUseTraceMetricsQuery.mockReturnValue({
      data: { data_points: [{ values: { [AggregationType.COUNT]: 5 } }] },
    });

    renderButton('exp-7');

    const copied = await openModalAndCopy();

    expect(copied).toContain('mlflow.set_experiment(experiment_id="exp-7")');
    expect(copied).toContain('mlflow.search_traces(max_results=20)');
    expect(copied).not.toContain('eval_dataset');
    expect(copied).not.toContain('predict_fn=predict');
  });
});

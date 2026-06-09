import { describe, beforeAll, beforeEach, afterEach, afterAll, jest, it, expect } from '@jest/globals';
import React from 'react';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AggregationType } from '@databricks/web-shared/model-trace-explorer';

import { EvaluationCodeSnippetButton } from './EvaluationCodeSnippetButton';

const mockUseTraceMetricsQuery = jest.fn();
jest.mock('../experiment-overview/hooks/useTraceMetricsQuery', () => ({
  __esModule: true,
  useTraceMetricsQuery: (...args: unknown[]) => mockUseTraceMetricsQuery(...args),
}));

const mockTraceMetricsCount = (count: number) =>
  mockUseTraceMetricsQuery.mockReturnValue({
    data: { data_points: [{ values: { [AggregationType.COUNT]: count } }] },
    isSuccess: true,
  });

const COPY_BUTTON_SELECTOR = '[data-component-id="mlflow.eval-runs.code-snippet-modal.copy-snippet"]';

describe('EvaluationCodeSnippetButton', () => {
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
        <EvaluationCodeSnippetButton experimentId={experimentId} />
      </DesignSystemProvider>,
    );

  // Click the copy-snippet button in the open modal and return what was written to the clipboard.
  const copyCurrentSnippet = async (): Promise<string> => {
    const copyButton = await waitFor(() => {
      const el = document.querySelector<HTMLElement>(COPY_BUTTON_SELECTOR);
      expect(el).not.toBeNull();
      return el!;
    });
    await userEvent.click(copyButton);

    expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
    return jest.mocked(navigator.clipboard.writeText).mock.calls[0][0] as string;
  };

  it('renders the dataset-based snippet and copies it when the experiment has no traces', async () => {
    mockTraceMetricsCount(0);

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'View code snippet' }));

    const copied = await copyCurrentSnippet();

    expect(copied).toContain('mlflow.set_experiment(experiment_id="exp-1")');
    expect(copied).toContain('eval_dataset = [{');
    expect(copied).toContain('predict_fn=predict');
    expect(copied).not.toContain('mlflow.search_traces');
  });

  it('renders the trace-based snippet and copies it when the experiment has traces', async () => {
    mockTraceMetricsCount(5);

    renderButton('exp-7');
    await userEvent.click(screen.getByRole('button', { name: 'View code snippet' }));

    const copied = await copyCurrentSnippet();

    expect(copied).toContain('mlflow.set_experiment(experiment_id="exp-7")');
    expect(copied).toContain('mlflow.search_traces(max_results=20)');
    expect(copied).not.toContain('eval_dataset');
    expect(copied).not.toContain('predict_fn=predict');
  });

  it('hides the snippet and copy button until the trace count query resolves', async () => {
    // Simulate the trace count query still in flight after the modal opens. Without this
    // guard, a fast copy-click would capture the default-dataset snippet for an experiment
    // that actually has traces.
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isSuccess: false });

    renderButton('exp-1');
    await userEvent.click(screen.getByRole('button', { name: 'View code snippet' }));

    expect(document.querySelector(COPY_BUTTON_SELECTOR)).toBeNull();
    expect(screen.queryByText(/eval_dataset = \[\{/)).not.toBeInTheDocument();
    expect(screen.queryByText(/mlflow\.search_traces/)).not.toBeInTheDocument();
  });
});

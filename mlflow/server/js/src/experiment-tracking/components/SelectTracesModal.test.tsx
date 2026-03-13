import { describe, jest, test, expect, beforeEach, beforeAll, afterEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { SelectTracesModal } from './SelectTracesModal';
import { useGenAiTraceTableRowSelection } from '../../shared/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import { useActiveEvaluation } from '../../shared/web-shared/genai-traces-table/hooks/useActiveEvaluation';
import { TracesV3Logs } from './experiment-page/components/traces-v3/TracesV3Logs';
import { TestRouter, setupTestRouter, testRoute, waitForRoutesToBeRendered } from '../../common/utils/RoutingTestUtils';

// Mock TracesV3Logs to keep this test simple
jest.mock('./experiment-page/components/traces-v3/TracesV3Logs', () => ({
  TracesV3Logs: jest.fn(),
}));

const testExperimentId = 'test-experiment-123';

describe('SelectTracesModal', () => {
  const mockWindowOpen = jest.fn();
  const { history } = setupTestRouter();

  beforeAll(() => {
    // Mock the TracesV3Logs component to return a simple mock traces table
    const MockTracesV3Logs = ({ experimentId }: { experimentId: string; endpointName: string }) => {
      const { rowSelection, setRowSelection } = useGenAiTraceTableRowSelection();
      const [, setSelectedEvaluationId] = useActiveEvaluation();

      // A few traces and checkboxes
      const mockTraces = [
        { id: 'trace-1', name: 'Trace 1' },
        { id: 'trace-2', name: 'Trace 2' },
        { id: 'trace-3', name: 'Trace 3' },
      ];

      return (
        <div data-testid="mock-traces-table">
          <div data-testid="experiment-id">{experimentId}</div>
          {mockTraces.map((trace) => (
            <label key={trace.id} data-testid={`trace-row-${trace.id}`}>
              <input
                type="checkbox"
                checked={Boolean(rowSelection[trace.id])}
                onChange={(e) => {
                  setRowSelection((prev: Record<string, boolean>) => ({
                    ...prev,
                    [trace.id]: e.target.checked,
                  }));
                }}
                data-testid={`checkbox-${trace.id}`}
              />
              <button data-testid={`view-trace-${trace.id}`} onClick={() => setSelectedEvaluationId(trace.id)}>
                {trace.name}
              </button>
            </label>
          ))}
        </div>
      );
    };
    jest.mocked(TracesV3Logs).mockImplementation(MockTracesV3Logs as any);

    // Mock window.open
    Object.defineProperty(window, 'open', {
      value: mockWindowOpen,
      writable: true,
    });
  });

  afterEach(() => {
    mockWindowOpen.mockClear();
  });

  const renderTestComponent = (props: {
    onClose?: () => void;
    onSuccess?: (traceIds: string[]) => void;
    maxTraceCount?: number;
  }) => {
    return render(
      <IntlProvider locale="en">
        <TestRouter
          history={history}
          routes={[
            testRoute(
              <DesignSystemProvider>
                <SelectTracesModal {...props} />
              </DesignSystemProvider>,
              '/experiments/:experimentId',
            ),
          ]}
          initialEntries={[`/experiments/${testExperimentId}`]}
        />
      </IntlProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('should call onSuccess with selected trace IDs when OK is clicked', async () => {
    const onSuccessMock = jest.fn();
    renderTestComponent({ onSuccess: onSuccessMock });
    await waitForRoutesToBeRendered();

    // Select two traces
    await userEvent.click(screen.getByTestId('checkbox-trace-1'));
    await userEvent.click(screen.getByTestId('checkbox-trace-2'));

    // Click the Select button
    const selectButton = screen.getByRole('button', { name: /select/i });
    await userEvent.click(selectButton);

    // Verify onSuccess was called with the selected trace IDs
    expect(onSuccessMock).toHaveBeenCalledTimes(1);
    expect(onSuccessMock).toHaveBeenCalledWith(['trace-1', 'trace-2']);
  });

  test('should disable OK button if all traces are deselected', async () => {
    renderTestComponent({});
    await waitForRoutesToBeRendered();

    // Select a trace
    await userEvent.click(screen.getByTestId('checkbox-trace-1'));

    // The Select button should be enabled
    let selectButton = screen.getByRole('button', { name: /select/i });
    expect(selectButton).toBeEnabled();

    // Deselect the trace
    await userEvent.click(screen.getByTestId('checkbox-trace-1'));

    // The Select button should be disabled again
    selectButton = screen.getByRole('button', { name: /select/i });
    expect(selectButton).toBeDisabled();
  });

  test('should disable OK button when max trace count is exceeded', async () => {
    renderTestComponent({ maxTraceCount: 2 });
    await waitForRoutesToBeRendered();

    // Select 3 traces (exceeds max of 2)
    await userEvent.click(screen.getByTestId('checkbox-trace-1'));
    await userEvent.click(screen.getByTestId('checkbox-trace-2'));
    await userEvent.click(screen.getByTestId('checkbox-trace-3'));

    const selectButton = screen.getByRole('button', { name: /select/i });
    expect(selectButton).toBeDisabled();
  });

  test('should enable OK button when selection is within max trace count', async () => {
    renderTestComponent({ maxTraceCount: 2 });
    await waitForRoutesToBeRendered();

    // Select 2 traces (within max of 2)
    await userEvent.click(screen.getByTestId('checkbox-trace-1'));
    await userEvent.click(screen.getByTestId('checkbox-trace-2'));

    const selectButton = screen.getByRole('button', { name: /select/i });
    expect(selectButton).toBeEnabled();
  });

  test('should open trace in new tab when clicking a trace', async () => {
    renderTestComponent({});
    await waitForRoutesToBeRendered();

    // Click to view a trace
    await userEvent.click(screen.getByTestId('view-trace-trace-1'));

    // Verify window.open was called with the correct URL
    expect(mockWindowOpen).toHaveBeenCalledTimes(1);
    expect(mockWindowOpen).toHaveBeenCalledWith(
      `/#/experiments/${testExperimentId}/traces?selectedEvaluationId=trace-1&startTimeLabel=LAST_7_DAYS`,
      '_blank',
    );
  });
});

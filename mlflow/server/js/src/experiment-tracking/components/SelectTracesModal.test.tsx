import { describe, jest, test, expect, beforeEach, beforeAll } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { SelectTracesModal } from './SelectTracesModal';
import { useGenAiTraceTableRowSelection } from '../../shared/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import { TracesV3Logs } from './experiment-page/components/traces-v3/TracesV3Logs';
import { TestRouter, testRoute } from '../../common/utils/RoutingTestUtils';

// Mock TracesV3Logs to keep this test simple
jest.mock('./experiment-page/components/traces-v3/TracesV3Logs', () => ({
  TracesV3Logs: jest.fn(),
}));

const testExperimentId = 'test-experiment-123';

describe('SelectTracesModal', () => {
  beforeAll(() => {
    // Mock the TracesV3Logs component to return a simple mock traces table
    const MockTracesV3Logs = ({ experimentId }: { experimentId: string; endpointName: string }) => {
      const { rowSelection, setRowSelection } = useGenAiTraceTableRowSelection();

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
              {trace.name}
            </label>
          ))}
        </div>
      );
    };
    jest.mocked(TracesV3Logs).mockImplementation(MockTracesV3Logs as any);
  });

  const renderTestComponent = (props: {
    onClose?: () => void;
    onSuccess?: (traceIds: string[]) => void;
    maxTraceCount?: number;
  }) => {
    return render(
      <IntlProvider locale="en">
        <TestRouter
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

    // Select 3 traces (exceeds max of 2)
    await userEvent.click(screen.getByTestId('checkbox-trace-1'));
    await userEvent.click(screen.getByTestId('checkbox-trace-2'));
    await userEvent.click(screen.getByTestId('checkbox-trace-3'));

    const selectButton = screen.getByRole('button', { name: /select/i });
    expect(selectButton).toBeDisabled();
  });

  test('should enable OK button when selection is within max trace count', async () => {
    renderTestComponent({ maxTraceCount: 2 });

    // Select 2 traces (within max of 2)
    await userEvent.click(screen.getByTestId('checkbox-trace-1'));
    await userEvent.click(screen.getByTestId('checkbox-trace-2'));

    const selectButton = screen.getByRole('button', { name: /select/i });
    expect(selectButton).toBeEnabled();
  });
});

import { describe, jest, test, expect, beforeEach, beforeAll } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { SelectSessionsModal } from './SelectSessionsModal';
import { useGenAiTraceTableRowSelection } from '../../shared/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import { GenAIChatSessionsTable, useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';
import { TestRouter, testRoute } from '../../common/utils/RoutingTestUtils';

// Mock GenAIChatSessionsTable to keep this test simple
jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/genai-traces-table')>(
    '@databricks/web-shared/genai-traces-table',
  ),
  GenAIChatSessionsTable: jest.fn(),
  useSearchMlflowTraces: jest.fn(),
}));

const testExperimentId = 'test-experiment-123';

describe('SelectSessionsModal', () => {
  beforeAll(() => {
    // Mock useSearchMlflowTraces to return empty data, we'll mock sessions directly
    jest.mocked(useSearchMlflowTraces).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    // Mock the GenAIChatSessionsTable component to return a simple mock sessions table
    const MockGenAIChatSessionsTable = () => {
      const { rowSelection, setRowSelection } = useGenAiTraceTableRowSelection();

      // A few sessions and checkboxes
      const mockSessions = [
        { id: 'session-1', name: 'Session 1' },
        { id: 'session-2', name: 'Session 2' },
        { id: 'session-3', name: 'Session 3' },
      ];

      return (
        <div data-testid="mock-sessions-table">
          {mockSessions.map((session) => (
            <label key={session.id} data-testid={`session-row-${session.id}`}>
              <input
                type="checkbox"
                checked={Boolean(rowSelection[session.id])}
                onChange={(e) => {
                  setRowSelection((prev: Record<string, boolean>) => ({
                    ...prev,
                    [session.id]: e.target.checked,
                  }));
                }}
                data-testid={`checkbox-${session.id}`}
              />
              {session.name}
            </label>
          ))}
        </div>
      );
    };
    jest.mocked(GenAIChatSessionsTable).mockImplementation(MockGenAIChatSessionsTable as any);
  });

  const renderTestComponent = (props: {
    onClose?: () => void;
    onSuccess?: (sessionIds: string[]) => void;
    maxSessionCount?: number;
  }) => {
    return render(
      <DesignSystemProvider>
        <IntlProvider locale="en">
          <TestRouter
            routes={[testRoute(<SelectSessionsModal {...props} />, '/experiments/:experimentId')]}
            initialEntries={[`/experiments/${testExperimentId}`]}
          />
        </IntlProvider>
      </DesignSystemProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Re-setup the mock for useSearchMlflowTraces after clearAllMocks
    jest.mocked(useSearchMlflowTraces).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);
  });

  test('should call onSuccess with selected session IDs when OK is clicked', async () => {
    const onSuccessMock = jest.fn();
    renderTestComponent({ onSuccess: onSuccessMock });

    // Select two sessions
    await userEvent.click(screen.getByTestId('checkbox-session-1'));
    await userEvent.click(screen.getByTestId('checkbox-session-2'));

    // Click the Select button
    const selectButton = screen.getByRole('button', { name: /select/i });
    await userEvent.click(selectButton);

    // Verify onSuccess was called with the selected session IDs
    expect(onSuccessMock).toHaveBeenCalledTimes(1);
    expect(onSuccessMock).toHaveBeenCalledWith(['session-1', 'session-2']);
  });

  test('should disable OK button if all sessions are deselected', async () => {
    renderTestComponent({});

    // Select a session
    await userEvent.click(screen.getByTestId('checkbox-session-1'));

    // The Select button should be enabled
    let selectButton = screen.getByRole('button', { name: /select/i });
    expect(selectButton).toBeEnabled();

    // Deselect the session
    await userEvent.click(screen.getByTestId('checkbox-session-1'));

    // The Select button should be disabled again
    selectButton = screen.getByRole('button', { name: /select/i });
    expect(selectButton).toBeDisabled();
  });

  test('should disable OK button when max session count is exceeded', async () => {
    renderTestComponent({ maxSessionCount: 2 });

    // Select 3 sessions (exceeds max of 2)
    await userEvent.click(screen.getByTestId('checkbox-session-1'));
    await userEvent.click(screen.getByTestId('checkbox-session-2'));
    await userEvent.click(screen.getByTestId('checkbox-session-3'));

    const selectButton = screen.getByRole('button', { name: /select/i });
    expect(selectButton).toBeDisabled();
  });

  test('should enable OK button when selection is within max session count', async () => {
    renderTestComponent({ maxSessionCount: 2 });

    // Select 2 sessions (within max of 2)
    await userEvent.click(screen.getByTestId('checkbox-session-1'));
    await userEvent.click(screen.getByTestId('checkbox-session-2'));

    const selectButton = screen.getByRole('button', { name: /select/i });
    expect(selectButton).toBeEnabled();
  });
});

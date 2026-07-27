import { jest, describe, beforeEach, afterEach, it, expect } from '@jest/globals';
import type { RowSelectionState } from '@tanstack/react-table';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen, waitFor, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

import { ExperimentEvaluationRunsTableActions } from './ExperimentEvaluationRunsTableActions';

const mockDeleteMutate = jest.fn();
// Capture the options passed to useDeleteRuns so tests can drive its onSuccess callback.
const mockDeleteRunsOptions: { onSuccess?: () => void } = {};
jest.mock('../../components/experiment-page/hooks/useDeleteRuns', () => ({
  __esModule: true,
  useDeleteRuns: ({ onSuccess }: { onSuccess: () => void }) => {
    mockDeleteRunsOptions.onSuccess = onSuccess;
    return { mutate: mockDeleteMutate, isLoading: false };
  },
}));

// The Actions Button renders a native `disabled` attribute; disable userEvent's
// pointer-events check so we can still fire clicks against it and assert the menu stays closed.
const setupUserEvent = () => userEvent.setup({ pointerEventsCheck: 0 });

const renderActions = (rowSelection: RowSelectionState) => {
  const setRowSelection = jest.fn();
  const refetchRuns = jest.fn();
  renderWithIntl(
    <DesignSystemProvider>
      <ExperimentEvaluationRunsTableActions
        rowSelection={rowSelection}
        setRowSelection={setRowSelection}
        refetchRuns={refetchRuns}
        onCompare={jest.fn()}
      />
    </DesignSystemProvider>,
  );
  return { setRowSelection, refetchRuns };
};

const getActionsButton = () => screen.getByRole('button', { name: 'Actions' });

describe('ExperimentEvaluationRunsTableActions', () => {
  let user: ReturnType<typeof setupUserEvent>;

  beforeEach(() => {
    user = setupUserEvent();
    mockDeleteRunsOptions.onSuccess = undefined;
    mockDeleteMutate.mockReset();
    // Simulate a successful delete by invoking the hook's onSuccess callback.
    mockDeleteMutate.mockImplementation(() => {
      mockDeleteRunsOptions.onSuccess?.();
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('disables the Actions trigger when no runs are selected', () => {
    renderActions({});
    expect(getActionsButton()).toBeDisabled();
  });

  it('does not open the menu when the disabled trigger is clicked with zero selections', async () => {
    renderActions({});

    await user.click(getActionsButton());

    // The Trigger is disabled, so no menu items should appear.
    expect(screen.queryByText('Delete runs')).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitem')).not.toBeInTheDocument();
  });

  it('enables the Actions trigger and opens the menu once a run is selected', async () => {
    renderActions({ 'run-1': true });

    const button = getActionsButton();
    expect(button).toBeEnabled();

    await user.click(button);

    expect(await screen.findByText('Delete runs')).toBeInTheDocument();
  });

  it('opens the delete confirmation modal when Delete runs is clicked with a selection', async () => {
    renderActions({ 'run-1': true });

    await user.click(getActionsButton());
    await user.click(await screen.findByText('Delete runs'));

    expect(await screen.findByRole('dialog', { name: /Delete 1 run/ })).toBeInTheDocument();
    // The mutation only fires on confirmation, not on opening the modal.
    expect(mockDeleteMutate).not.toHaveBeenCalled();
  });

  it('deletes the selected runs and clears the selection when the modal is confirmed', async () => {
    const { setRowSelection, refetchRuns } = renderActions({ 'run-1': true, 'run-2': true });

    await user.click(getActionsButton());
    await user.click(await screen.findByText('Delete runs'));

    const dialog = await screen.findByRole('dialog', { name: /Delete 2 runs/ });
    await user.click(within(dialog).getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(mockDeleteMutate).toHaveBeenCalledWith({ runUuids: ['run-1', 'run-2'] });
    });

    // On success the component refetches and clears the row selection.
    expect(refetchRuns).toHaveBeenCalledTimes(1);
    expect(setRowSelection).toHaveBeenCalledWith({});
  });
});

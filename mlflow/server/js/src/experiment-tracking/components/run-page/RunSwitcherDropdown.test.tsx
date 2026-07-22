import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { renderWithIntl, screen, within, cleanup, waitFor } from '../../../common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { DesignSystemProvider } from '@databricks/design-system';
import userEvent from '@testing-library/user-event';
import { RunSwitcherDropdown } from './RunSwitcherDropdown';
import { MlflowService } from '../../sdk/MlflowService';
import type { RunEntity } from '../../types';
import Routes from '../../routes';

jest.mock('../../sdk/MlflowService', () => ({
  MlflowService: {
    searchRuns: jest.fn(),
  },
}));

const mockNavigate = jest.fn();
jest.mock('../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/RoutingUtils')>('../../../common/utils/RoutingUtils'),
  useNavigate: () => mockNavigate,
}));

const createRun = (runUuid: string, runName: string): RunEntity => ({
  info: {
    runUuid,
    runName,
    experimentId: 'exp-id',
    artifactUri: '',
    endTime: 0,
    lifecycleStage: 'active',
    startTime: 0,
    status: 'FINISHED',
  },
  data: { metrics: [], params: [], tags: [] },
});

interface RenderProps {
  comparisonRunUuids?: string[];
  onCompareRun?: (run: RunEntity) => void;
  onClearComparisons?: () => void;
}

const renderComponent = (props: RenderProps = {}) =>
  renderWithIntl(
    <MemoryRouter>
      <DesignSystemProvider>
        <RunSwitcherDropdown experimentId="exp-id" currentRunUuid="run-1" activeTab="model-metrics" {...props} />
      </DesignSystemProvider>
    </MemoryRouter>,
  );

const openDropdown = async () => {
  await userEvent.click(screen.getByRole('button', { name: 'Switch run' }));
  await screen.findByPlaceholderText('Search runs');
};

describe('RunSwitcherDropdown', () => {
  beforeEach(() => {
    jest.mocked(MlflowService.searchRuns).mockReset();
    mockNavigate.mockClear();
  });

  test('renders the trigger button', () => {
    renderComponent();
    expect(screen.getByRole('button', { name: 'Switch run' })).toBeInTheDocument();
  });

  test('fetches runs from the experiment on first open and displays them', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [createRun('run-1', 'Run 1'), createRun('run-2', 'Run 2')],
    });

    renderComponent();
    await openDropdown();

    expect(MlflowService.searchRuns).toHaveBeenCalledWith({
      experiment_ids: ['exp-id'],
      max_results: 200,
    });
    expect(screen.getByText('Run 1')).toBeInTheDocument();
    expect(screen.getByText('Run 2')).toBeInTheDocument();
  });

  test('does not re-fetch when opened a second time', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [createRun('run-1', 'Run 1')],
    });

    renderComponent();
    await openDropdown();

    // Close by clicking the trigger again (Escape is swallowed by the search input's onKeyDown)
    await userEvent.click(screen.getByRole('button', { name: 'Switch run' }));
    await waitFor(() => expect(screen.queryByPlaceholderText('Search runs')).not.toBeInTheDocument());

    await openDropdown();

    expect(jest.mocked(MlflowService.searchRuns)).toHaveBeenCalledTimes(1);
  });

  test('shows loading state while runs are being fetched', async () => {
    jest.mocked(MlflowService.searchRuns).mockImplementationOnce(() => new Promise(() => {}));
    renderComponent();
    await openDropdown();
    expect(screen.getByText('Loading…')).toBeInTheDocument();
  });

  test('shows empty list when the fetch fails', async () => {
    jest.mocked(MlflowService.searchRuns).mockRejectedValueOnce(new Error('Network error'));
    renderComponent();
    await openDropdown();
    await waitFor(() => expect(screen.getByText('No runs found')).toBeInTheDocument());
  });

  test('filters the run list by search input and shows "No runs found" when nothing matches', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [createRun('run-1', 'Alpha run'), createRun('run-2', 'Beta run'), createRun('run-3', 'Gamma run')],
    });

    renderComponent();
    await openDropdown();

    await userEvent.type(screen.getByPlaceholderText('Search runs'), 'Alpha');

    expect(screen.getByText('Alpha run')).toBeInTheDocument();
    expect(screen.queryByText('Beta run')).not.toBeInTheDocument();
    expect(screen.queryByText('Gamma run')).not.toBeInTheDocument();

    await userEvent.clear(screen.getByPlaceholderText('Search runs'));
    await userEvent.type(screen.getByPlaceholderText('Search runs'), 'zzz-no-match');

    expect(screen.getByText('No runs found')).toBeInTheDocument();
  });

  test('navigates to the selected run keeping the active tab', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [createRun('run-1', 'Run 1'), createRun('run-2', 'Run 2')],
    });

    renderComponent();
    await openDropdown();

    await userEvent.click(screen.getByText('Run 2'));

    expect(mockNavigate).toHaveBeenCalledWith(Routes.getRunPageTabRoute('exp-id', 'run-2', 'model-metrics'));
  });

  test('shows compare button for non-current runs only when onCompareRun is provided', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [createRun('run-1', 'Run 1'), createRun('run-2', 'Run 2'), createRun('run-3', 'Run 3')],
    });

    renderComponent({ onCompareRun: jest.fn() });
    await openDropdown();

    const run1Item = screen.getByRole('menuitemcheckbox', { name: /Run 1/ });
    const run2Item = screen.getByRole('menuitemcheckbox', { name: /Run 2/ });

    // Current run has no compare button; non-current runs do
    expect(within(run1Item).queryByRole('button')).not.toBeInTheDocument();
    expect(within(run2Item).getByRole('button')).toBeInTheDocument();
  });

  test('clicking the compare button calls onCompareRun without triggering navigation', async () => {
    const onCompareRun = jest.fn();
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [createRun('run-1', 'Run 1'), createRun('run-2', 'Run 2')],
    });

    renderComponent({ onCompareRun });
    await openDropdown();

    const run2Item = screen.getByRole('menuitemcheckbox', { name: /Run 2/ });
    await userEvent.click(within(run2Item).getByRole('button'));

    expect(onCompareRun).toHaveBeenCalledWith(createRun('run-2', 'Run 2'));
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  test('disables compare button at the 5-run cap but not for runs already in comparison', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [
        createRun('run-1', 'Run 1'),
        createRun('run-2', 'Run 2'),
        createRun('run-3', 'Run 3'),
        createRun('run-4', 'Run 4'),
        createRun('run-5', 'Run 5'),
        createRun('run-6', 'Run 6'),
        createRun('run-7', 'Run 7'),
      ],
    });

    renderComponent({
      onCompareRun: jest.fn(),
      comparisonRunUuids: ['run-2', 'run-3', 'run-4', 'run-5', 'run-6'],
    });
    await openDropdown();

    // run-7 is not in comparison and cap is reached — should be disabled
    expect(within(screen.getByRole('menuitemcheckbox', { name: /Run 7/ })).getByRole('button')).toBeDisabled();
    // run-2 is already pinned — can still be toggled off
    expect(within(screen.getByRole('menuitemcheckbox', { name: /Run 2/ })).getByRole('button')).not.toBeDisabled();
  });

  test('renders comparison label as UUID for one run, count for multiple, and hidden when empty', () => {
    // Single comparison: shows UUID as fallback before runs are loaded
    renderComponent({ comparisonRunUuids: ['run-2'], onClearComparisons: jest.fn() });
    expect(screen.getByText('vs')).toBeInTheDocument();
    expect(screen.getByText('run-2')).toBeInTheDocument();
    cleanup();

    // Multiple comparisons: shows count
    renderComponent({ comparisonRunUuids: ['run-2', 'run-3'], onClearComparisons: jest.fn() });
    expect(screen.getByText('vs')).toBeInTheDocument();
    expect(screen.getByText('2 runs')).toBeInTheDocument();
    cleanup();

    // No comparisons: label is absent
    renderComponent();
    expect(screen.queryByText('vs')).not.toBeInTheDocument();
  });

  test('calls onClearComparisons when the clear button is clicked', async () => {
    const onClearComparisons = jest.fn();
    renderComponent({ comparisonRunUuids: ['run-2'], onClearComparisons });
    await userEvent.click(screen.getByRole('button', { name: 'Clear comparison' }));
    expect(onClearComparisons).toHaveBeenCalledTimes(1);
  });
});

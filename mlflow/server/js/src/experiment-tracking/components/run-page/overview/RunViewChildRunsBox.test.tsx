import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { renderWithIntl, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { DesignSystemProvider } from '@databricks/design-system';
import { RunViewChildRunsBox } from './RunViewChildRunsBox';
import { MlflowService } from '../../../sdk/MlflowService';
import userEvent from '@testing-library/user-event';
import type { RunInfoEntity } from '../../../types';

jest.mock('../../../sdk/MlflowService', () => ({
  MlflowService: {
    searchRuns: jest.fn(),
  },
}));

const experimentId = 'exp-id';
const parentRunUuid = 'parent-run';

const createRunInfo = (runUuid: string, runName: string): RunInfoEntity => ({
  artifactUri: '',
  endTime: 0,
  experimentId,
  lifecycleStage: 'active',
  runUuid,
  runName,
  startTime: 0,
  status: 'FINISHED',
});

describe('RunViewChildRunsBox', () => {
  const renderComponent = () =>
    renderWithIntl(
      <MemoryRouter>
        <DesignSystemProvider>
          <RunViewChildRunsBox runUuid={parentRunUuid} experimentId={experimentId} />
        </DesignSystemProvider>
      </MemoryRouter>,
    );

  beforeEach(() => {
    jest.mocked(MlflowService.searchRuns).mockReset();
  });

  test('renders loading state and displays child runs', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [{ info: createRunInfo('child-1', 'Child run 1'), data: { metrics: [], params: [], tags: [] } }],
      next_page_token: undefined,
    });

    renderComponent();

    expect(screen.getByText('Child runs loading')).toBeInTheDocument();

    // Check for the label
    expect(await screen.findByText('Child runs')).toBeInTheDocument();

    const link = await screen.findByRole('link', { name: 'Child run 1' });
    expect(link).toHaveAttribute('href', `/experiments/${experimentId}/runs/child-1`);
    expect(screen.queryByText('Child runs loading')).not.toBeInTheDocument();
  });

  test('renders error message when API call fails', async () => {
    jest.mocked(MlflowService.searchRuns).mockRejectedValueOnce(new Error('boom'));

    renderComponent();

    expect(await screen.findByText('Failed to load child runs')).toBeInTheDocument();
    expect(screen.getByText('Child runs')).toBeInTheDocument();
  });

  test('renders nothing when no child runs are returned', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({ runs: [], next_page_token: undefined });

    renderComponent();

    expect(screen.getByText('Child runs loading')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.queryByText('Child runs loading')).not.toBeInTheDocument();
    });

    expect(screen.queryByText('Child runs')).not.toBeInTheDocument();
  });

  test('loads more runs when clicking the Load more button', async () => {
    jest
      .mocked(MlflowService.searchRuns)
      .mockResolvedValueOnce({
        runs: [{ info: createRunInfo('child-1', 'Child run 1'), data: { metrics: [], params: [], tags: [] } }],
        next_page_token: 'next-token',
      })
      .mockResolvedValueOnce({
        runs: [{ info: createRunInfo('child-2', 'Child run 2'), data: { metrics: [], params: [], tags: [] } }],
        next_page_token: undefined,
      });

    renderComponent();

    expect(await screen.findByText('Child runs')).toBeInTheDocument();
    expect(await screen.findByRole('link', { name: 'Child run 1' })).toBeInTheDocument();

    const loadMore = screen.getByRole('button', { name: 'Load more' });
    await userEvent.click(loadMore);

    expect(await screen.findByRole('link', { name: 'Child run 2' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load more' })).not.toBeInTheDocument();
    expect(jest.mocked(MlflowService.searchRuns)).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ page_token: 'next-token' }),
    );
  });
});

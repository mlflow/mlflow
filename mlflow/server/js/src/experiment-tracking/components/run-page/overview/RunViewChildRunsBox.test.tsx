import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { DesignSystemProvider } from '@databricks/design-system';
import { RunViewChildRunsBox } from './RunViewChildRunsBox';
import { MlflowService } from '../../../sdk/MlflowService';
import { EXPERIMENT_PARENT_ID_TAG } from '../../experiment-page/utils/experimentPage.common-utils';

jest.mock('../../../sdk/MlflowService', () => ({
  MlflowService: {
    searchRuns: jest.fn(),
  },
}));

const experimentId = 'exp-id';
const parentRunUuid = 'parent-run';

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

  test('renders loading state and displays link to experiment page', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({
      runs: [
        {
          info: {
            artifactUri: '',
            endTime: 0,
            experimentId,
            lifecycleStage: 'active',
            runUuid: parentRunUuid,
            runName: 'Parent run',
            startTime: 0,
            status: 'FINISHED',
          },
          data: { tags: [{ key: EXPERIMENT_PARENT_ID_TAG, value: parentRunUuid }], params: [], metrics: [] },
        },
      ],
      next_page_token: undefined,
    });

    renderComponent();

    expect(screen.getByText('Child runs loading')).toBeInTheDocument();

    const expectedFilter = encodeURIComponent(`tags.\`${EXPERIMENT_PARENT_ID_TAG}\` = '${parentRunUuid}'`);
    const link = await screen.findByRole('link', { name: 'View all child runs in the experiment page' });
    expect(link).toHaveAttribute('href', `/experiments/${experimentId}/runs?searchFilter=${expectedFilter}`);
    expect(screen.queryByText('Child runs loading')).not.toBeInTheDocument();
  });

  test('renders error message when API call fails', async () => {
    jest.mocked(MlflowService.searchRuns).mockRejectedValueOnce(new Error('boom'));

    renderComponent();

    expect(await screen.findByText('Failed to load child runs')).toBeInTheDocument();
  });

  test('renders dash when no child runs are returned', async () => {
    jest.mocked(MlflowService.searchRuns).mockResolvedValueOnce({ runs: [], next_page_token: undefined });

    renderComponent();

    expect(await screen.findByText('—')).toBeInTheDocument();
  });
});

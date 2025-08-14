import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { RunViewChildRunsBox } from '../overview/RunViewChildRunsBox';
import { MlflowService } from '../../../sdk/MlflowService';

describe('RunViewChildRunsBox', () => {
  const parentRunUuid = 'parent-run-uuid';
  const experimentId = 'exp-1';

  const renderComponent = () =>
    renderWithIntl(
      <DesignSystemProvider>
        <MemoryRouter>
          <RunViewChildRunsBox parentRunUuid={parentRunUuid} experimentId={experimentId} />
        </MemoryRouter>
      </DesignSystemProvider>,
    );

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('renders list of child runs when present', async () => {
    jest.spyOn(MlflowService, 'searchRuns').mockResolvedValue({
      runs: [
        {
          info: {
            runUuid: 'child-1',
            runName: 'Child Run 1',
            status: 'FINISHED',
            experimentId: experimentId,
            lifecycleStage: 'active',
            startTime: 1718332800,
            endTime: 1718332800,
            artifactUri: 'artifact-uri',
          },
          data: {
            tags: [],
            params: [],
            metrics: [],
          },
        },
        {
          info: {
            runUuid: 'child-2',
            runName: 'Child Run 2',
            status: 'RUNNING',
            experimentId: experimentId,
            lifecycleStage: 'active',
            startTime: 1718332800,
            endTime: 1718332800,
            artifactUri: 'artifact-uri',
          },
          data: {
            tags: [{ key: 'mlflow.parentRunId', value: parentRunUuid }],
            params: [],
            metrics: [],
          },
        },
      ],
    });

    renderComponent();

    expect(await screen.findByText('Found 2 child runs')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Child Run 1' })).toBeInTheDocument();
    expect(screen.getByText('(FINISHED)')).toBeInTheDocument();

    expect(MlflowService.searchRuns).toHaveBeenCalledWith(
      expect.objectContaining({
        experiment_ids: [experimentId],
        filter: `tags.mlflow.parentRunId = '${parentRunUuid}'`,
      }),
    );
  });

  test('renders message when no child runs found', async () => {
    jest.spyOn(MlflowService, 'searchRuns').mockResolvedValue({ runs: [] });

    renderComponent();

    expect(await screen.findByText('No child runs found')).toBeInTheDocument();
  });

  test('renders error message when fetching fails', async () => {
    jest.spyOn(MlflowService, 'searchRuns').mockRejectedValue(new Error('Network error'));

    renderComponent();

    expect(await screen.findByText('Failed to load child runs')).toBeInTheDocument();
  });
});

import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { LoggedModelProto } from '../../types';
import { ExperimentLoggedModelDetailsTraces } from './ExperimentLoggedModelDetailsTraces';
import { isExperimentLoggedModelsUIEnabled } from '../../../common/utils/FeatureUtils';

jest.setTimeout(90000); // Larger timeout for integration testing (table rendering)

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  isExperimentLoggedModelsUIEnabled: jest.fn(),
}));

describe('ExperimentLoggedModelDetailsTraces integration test', () => {
  const { history } = setupTestRouter();
  const server = setupServer(
    rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => {
      const { searchParams } = req.url;
      if (
        searchParams.getAll('experiment_ids').includes('test-experiment') &&
        searchParams.get('filter')?.includes("request_metadata.`mlflow.modelId`='m-test-model-id'")
      ) {
        return res(
          ctx.json({
            traces: [
              {
                request_id: 'tr-12345',
                experiment_id: 'test-experiment',
                timestamp_ms: 1730722066662,
                execution_time_ms: 3000,
                status: 'OK',
                request_metadata: [{ key: 'mlflow.modelId', value: 'm-test-model-id' }],
                tags: [{ key: 'mlflow.traceName', value: 'RunnableSequence' }],
              },
            ],
          }),
        );
      }

      return res(ctx.json({ traces: [] }));
    }),
  );

  const renderTestComponent = (loggedModel: LoggedModelProto) => {
    return render(<ExperimentLoggedModelDetailsTraces loggedModel={loggedModel} />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <TestRouter routes={[testRoute(<>{children}</>)]} history={history} />
        </IntlProvider>
      ),
    });
  };

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(() => {
    jest.mocked(isExperimentLoggedModelsUIEnabled).mockReturnValue(true);
  });

  test('should fetch and display table of traces', async () => {
    renderTestComponent({
      info: {
        experiment_id: 'test-experiment',
        model_id: 'm-test-model-id',
      },
    });

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search traces')).toBeInTheDocument();
      expect(screen.getByRole('cell', { name: 'tr-12345' })).toBeInTheDocument();
    });
  });
  test('should display empty table when model contains no traces', async () => {
    renderTestComponent({
      info: {
        experiment_id: 'test-experiment',
        model_id: 'm-some-other-model-id-with-no-traces',
      },
    });

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'No traces recorded' })).toBeInTheDocument();
    });
  });
});

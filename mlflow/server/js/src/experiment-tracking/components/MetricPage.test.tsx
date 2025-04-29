import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { Provider } from 'react-redux';
import { applyMiddleware, combineReducers, createStore } from 'redux';
import promiseMiddleware from 'redux-promise-middleware';
import thunk from 'redux-thunk';
import { setupTestRouter, testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { setupServer } from '../../common/utils/setup-msw';
import { render, screen, waitFor } from '../../common/utils/TestUtils.react18';
import MetricPage from './MetricPage';
import { DesignSystemProvider } from '@databricks/design-system';
import { apis } from '../reducers/Reducers';

// We care only about the MetricPage component, so we mock the inner components
jest.mock('./MetricView', () => ({
  MetricView: () => <div>view loaded successfully</div>,
}));

describe('MetricPage', () => {
  const { history } = setupTestRouter();
  const server = setupServer();

  const mockSuccessExperimentsResponse = () => {
    return rest.get('/ajax-api/2.0/mlflow/experiments/get', (req, res, ctx) => {
      return res(ctx.json({ experiment: { experiment_id: 123, name: 'experiment1' } }));
    });
  };

  const mockSuccessRunsResponse = () =>
    rest.get('/ajax-api/2.0/mlflow/runs/get', (req, res, ctx) => {
      return res(ctx.json({ run: { info: { run_id: 'run123', run_uuid: 'run123', runName: 'run123' }, data: {} } }));
    });

  const renderTestComponent = (
    initialRoute = '/?runs=%5B"run123"%5D&metric="test_mean_absolute_error"&experiments=%5B"123"%5D&plot_metric_keys=%5B"test_mean_absolute_error"%5D&plot_layout=%7B"autosize"%3Atrue%2C"xaxis"%3A%7B%7D%2C"yaxis"%3A%7B%7D%7D&x_axis=relative&y_axis_scale=linear&line_smoothness=1&show_point=false&deselected_curves=%5B%5D&last_linear_y_axis_range=%5B%5D&o=123',
  ) => {
    const store = createStore(
      combineReducers({
        apis,
      }),
      applyMiddleware(thunk, promiseMiddleware()),
    );

    render(<MetricPage />, {
      wrapper: ({ children }) => (
        <Provider store={store}>
          <DesignSystemProvider>
            <IntlProvider locale="en">
              <TestRouter
                routes={[testRoute(<>{children}</>, '/')]}
                history={history}
                initialEntries={[initialRoute]}
              />
            </IntlProvider>
          </DesignSystemProvider>
        </Provider>
      ),
    });
  };
  test('should render mocked view content when everything loads successfully', async () => {
    server.use(mockSuccessExperimentsResponse(), mockSuccessRunsResponse());
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText('view loaded successfully')).toBeVisible();
    });
  });

  describe('Error states', () => {
    beforeEach(() => {
      // Avoid flooding console with expected errors
      jest.spyOn(console, 'error').mockImplementation(() => {});
    });
    afterEach(() => {
      jest.restoreAllMocks();
    });

    test('should render error when URL is malformed', async () => {
      renderTestComponent('/?blah=bleh');

      await waitFor(() => {
        expect(screen.getByText('Error during metric page load: invalid URL')).toBeVisible();
      });
    });

    test('should render relevant error when runs fail to load', async () => {
      server.use(
        mockSuccessExperimentsResponse(),
        rest.get('/ajax-api/2.0/mlflow/runs/get', (req, res, ctx) => {
          return res(ctx.json({ error_code: 'RESOURCE_DOES_NOT_EXIST', message: 'Run not found' }), ctx.status(404));
        }),
      );

      renderTestComponent();

      await waitFor(() => {
        expect(screen.getByText('Error while loading metric page')).toBeVisible();
      });

      expect(screen.getByText('Run not found')).toBeVisible();
    });

    test('should render relevant error when experiment fail to load', async () => {
      server.use(
        mockSuccessRunsResponse(),
        rest.get('/ajax-api/2.0/mlflow/experiments/get', (req, res, ctx) => {
          return res(
            ctx.json({ error_code: 'RESOURCE_DOES_NOT_EXIST', message: 'Experiment not found' }),
            ctx.status(404),
          );
        }),
      );

      renderTestComponent();

      await waitFor(() => {
        expect(screen.getByText('Error while loading metric page')).toBeVisible();
      });

      expect(screen.getByText('Experiment not found')).toBeVisible();
    });
  });
});

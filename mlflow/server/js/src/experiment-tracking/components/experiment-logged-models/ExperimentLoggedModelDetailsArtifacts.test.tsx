import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { Provider } from 'react-redux';
import { applyMiddleware, combineReducers, compose, createStore } from 'redux';
import promiseMiddleware from 'redux-promise-middleware';
import thunk from 'redux-thunk';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, waitFor } from '../../../common/utils/TestUtils.react18';
import { apis, artifactsByRunUuid } from '../../reducers/Reducers';
import { ExperimentLoggedModelDetailsArtifacts } from './ExperimentLoggedModelDetailsArtifacts';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';

describe('ExperimentLoggedModelDetailsArtifacts integration test', () => {
  const { history } = setupTestRouter();
  const server = setupServer(
    rest.get('/ajax-api/2.0/mlflow/logged-models/:modelId/artifacts/directories', (req, res, ctx) =>
      res(
        ctx.json({
          root_uri: 'dbfs:/databricks/mlflow-tracking/123/logged_models/test-model-id/artifacts',
          files: [
            {
              path: 'conda.yaml',
              is_dir: false,
              file_size: 123,
            },
            {
              path: 'requirements.txt',
              is_dir: false,
              file_size: 234,
            },
          ],
        }),
      ),
    ),
    rest.get('/ajax-api/2.0/mlflow/logged-models/:modelId/artifacts/files', (req, res, ctx) =>
      res(ctx.text('this is text file content of ' + req.url.searchParams.get('artifact_file_path'))),
    ),
    rest.get('/get-artifact', (req, res, ctx) =>
      res(ctx.text('this is text file content of ' + req.url.searchParams.get('path'))),
    ),
  );

  const renderTestComponent = () => {
    const loggedModel = {
      info: {
        model_id: 'test-model-id',
        artifact_uri: 'dbfs:/databricks/mlflow-tracking/123/logged_models/test-model-id/artifacts',
      },
    };

    const store = createStore(
      combineReducers({
        entities: combineReducers({
          artifactsByRunUuid,
          modelVersionsByModel: () => ({}),
        }),
        apis,
      }),
      {},
      compose(applyMiddleware(thunk, promiseMiddleware())),
    );

    return render(<ExperimentLoggedModelDetailsArtifacts loggedModel={loggedModel} />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <Provider store={store}>
            <TestRouter routes={[testRoute(<>{children}</>)]} history={history} />
          </Provider>
        </IntlProvider>
      ),
    });
  };

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  test('should render list of artifacts and display file contents', async () => {
    const { getByText } = renderTestComponent();

    await waitFor(() => {
      expect(getByText('requirements.txt')).toBeInTheDocument();
      expect(getByText('conda.yaml')).toBeInTheDocument();
    });

    await userEvent.click(getByText('requirements.txt'));

    await waitFor(() => {
      expect(getByText('this is text file content of requirements.txt')).toBeInTheDocument();
    });
  });
});

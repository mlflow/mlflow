import { afterAll, afterEach, jest, describe, beforeAll, test, expect } from '@jest/globals';

import { DesignSystemProvider } from '@databricks/design-system';
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
import { setActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { getWorkspacesEnabledSync } from '../../hooks/useServerInfo';

jest.mock('../../hooks/useServerInfo', () => ({
  ...jest.requireActual<typeof import('../../hooks/useServerInfo')>('../../hooks/useServerInfo'),
  getWorkspacesEnabledSync: jest.fn(),
}));

const getWorkspacesEnabledSyncMock = jest.mocked(getWorkspacesEnabledSync);

describe('ExperimentLoggedModelDetailsArtifacts integration test', () => {
  const { history } = setupTestRouter();
  let capturedRequests: { url: string; workspaceHeader: string | null; storageHeader: string | null }[] = [];
  const server = setupServer(
    rest.get(/\/?ajax-api\/2\.0\/mlflow\/logged-models\/[^/]+\/artifacts\/directories/, (req, res, ctx) => {
      capturedRequests.push({
        url: req.url.toString(),
        workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE'),
        storageHeader: req.headers.get('x-storage-header'),
      });
      return res(
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
      );
    }),
    rest.post(
      /\/?ajax-api\/2\.0\/mlflow\/logged-models\/[^/]+\/artifacts\/credentials-for-download/,
      async (req, res, ctx) => {
        capturedRequests.push({
          url: req.url.toString(),
          workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE'),
          storageHeader: req.headers.get('x-storage-header'),
        });
        const body = (await req.json()) as { paths: string[] };
        const path = body.paths[0];
        return res(
          ctx.json({
            credentials: [
              {
                credential_info: {
                  type: 'AWS_PRESIGNED_URL',
                  signed_uri: `https://storage.example/artifacts/${encodeURIComponent(path)}`,
                  path,
                  headers: [{ name: 'x-storage-header', value: 'required' }],
                },
              },
            ],
          }),
        );
      },
    ),
    rest.get(/^https:\/\/storage\.example\/artifacts\//, (req, res, ctx) => {
      capturedRequests.push({
        url: req.url.toString(),
        workspaceHeader: req.headers.get('X-MLFLOW-WORKSPACE'),
        storageHeader: req.headers.get('x-storage-header'),
      });
      const path = decodeURIComponent(req.url.pathname.split('/').pop() ?? '');
      return res(ctx.text('this is text file content of ' + path));
    }),
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
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <Provider store={store}>
              <TestRouter routes={[testRoute(<>{children}</>)]} history={history} />
            </Provider>
          </IntlProvider>
        </DesignSystemProvider>
      ),
    });
  };

  beforeAll(() => {
    getWorkspacesEnabledSyncMock.mockReturnValue(true);
    setActiveWorkspace('team-a');
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  afterAll(() => {
    jest.restoreAllMocks();
    setActiveWorkspace(null);
    server.close();
  });

  afterEach(() => {
    capturedRequests = [];
    server.resetHandlers();
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

    const trackingServerRequests = capturedRequests.filter((req) => !req.url.startsWith('https://storage.example/'));
    expect(trackingServerRequests.length).toBeGreaterThan(0);
    for (const req of trackingServerRequests) {
      expect(req.workspaceHeader).toBe('team-a');
    }
    expect(
      capturedRequests.some((req) => req.url.includes('/artifacts/files') || req.url.includes('/get-artifact')),
    ).toBe(false);
    expect(capturedRequests.find((req) => req.url.startsWith('https://storage.example/'))).toEqual(
      expect.objectContaining({ workspaceHeader: null, storageHeader: 'required' }),
    );
  });
});

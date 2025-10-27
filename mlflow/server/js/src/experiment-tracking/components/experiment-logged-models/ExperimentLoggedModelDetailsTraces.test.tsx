import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import type { LoggedModelProto } from '../../types';
import { ExperimentLoggedModelDetailsTraces } from './ExperimentLoggedModelDetailsTraces';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { DesignSystemProvider } from '@databricks/design-system';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(90000); // Larger timeout for integration testing (table rendering)

describe('ExperimentLoggedModelDetailsTraces integration test', () => {
  const queryClient = new QueryClient();
  const { history } = setupTestRouter();
  const server = setupServer(
    rest.post('/ajax-api/3.0/mlflow/traces/search', (req, res, ctx) => {
      return res(
        ctx.json({
          traces: [
            {
              trace_id: 'trace_1',
              request: '{"input": "value"}',
              response: '{"output": "value"}',
              trace_metadata: {
                user_id: 'user123',
                environment: 'production',
                'mlflow.internal.key': 'internal_value',
              },
            },
          ],
          next_page_token: undefined,
        }),
      );
    }),
  );

  const renderTestComponent = (loggedModel: LoggedModelProto) => {
    return render(<ExperimentLoggedModelDetailsTraces loggedModel={loggedModel} />, {
      wrapper: ({ children }) => (
        <DesignSystemProvider>
          <QueryClientProvider client={queryClient}>
            <IntlProvider locale="en">
              <TestRouter
                routes={[testRoute(<>{children}</>, '/experiments/:experimentId/models/:loggedModelId')]}
                history={history}
                initialEntries={[
                  `/experiments/${loggedModel.info?.experiment_id}/models/${loggedModel.info?.model_id}`,
                ]}
              />
            </IntlProvider>
          </QueryClientProvider>
        </DesignSystemProvider>
      ),
    });
  };

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  test('should fetch and display table of traces', async () => {
    renderTestComponent({
      info: {
        experiment_id: 'test-experiment',
        model_id: 'm-test-model-id',
      },
    });

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search traces by request')).toBeInTheDocument();
    });
  });
  test('should display quickstart when model contains no traces', async () => {
    renderTestComponent({
      info: {
        experiment_id: 'test-experiment',
        model_id: 'm-some-other-model-id-with-no-traces',
      },
    });

    await waitFor(() => {
      expect(document.body.textContent).not.toBe('');
    });
  });
});

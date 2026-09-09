import { describe, expect, test } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../common/utils/setup-msw';
import { renderWithIntl, screen, waitFor } from '../../common/utils/TestUtils.react18';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../common/utils/reactQueryHooks';
import Routes from '../../experiment-tracking/routes';
import { ModelVersionSourceModelLink } from './ModelVersionSourceModelLink';

const loggedModelId = 'm-1bc96c322a8441c7a5af8de3df2cc4ce';

describe('ModelVersionSourceModelLink', () => {
  const server = setupServer();

  const renderComponent = () => {
    const queryClient = new QueryClient();
    renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <ModelVersionSourceModelLink loggedModelId={loggedModelId} />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    return queryClient;
  };

  test('links to the logged model page using its name', async () => {
    server.use(
      rest.get(`/ajax-api/2.0/mlflow/logged-models/${loggedModelId}`, (req, res, ctx) =>
        res(ctx.json({ model: { info: { model_id: loggedModelId, experiment_id: '123', name: 'iris_model' } } })),
      ),
    );
    renderComponent();

    const link = await screen.findByTestId('source-model-link');
    expect(link).toHaveTextContent('iris_model');
    expect(link).toHaveAttribute(
      'href',
      expect.stringContaining(Routes.getExperimentLoggedModelDetailsPageRoute('123', loggedModelId)),
    );
  });

  test('falls back to the plain model ID when the logged model cannot be fetched', async () => {
    server.use(
      rest.get(`/ajax-api/2.0/mlflow/logged-models/${loggedModelId}`, (req, res, ctx) =>
        res(ctx.status(404), ctx.json({ error_code: 'RESOURCE_DOES_NOT_EXIST' })),
      ),
    );
    const queryClient = renderComponent();

    // The plain ID is also shown while loading, so wait for the request to fail before asserting
    await waitFor(() => expect(queryClient.getQueryCache().getAll()[0]?.state.status).toBe('error'));
    expect(screen.getByTestId('source-model-id')).toHaveTextContent(loggedModelId);
    expect(screen.queryByTestId('source-model-link')).not.toBeInTheDocument();
  });
});

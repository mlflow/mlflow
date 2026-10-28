import { describe, expect, test } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../../common/utils/setup-msw';
import { renderWithIntl, screen } from '../../common/utils/TestUtils.react18';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../../common/utils/reactQueryHooks';
import Routes from '../../experiment-tracking/routes';
import { ModelVersionSourceModelLink } from './ModelVersionSourceModelLink';

const loggedModelId = 'm-1bc96c322a8441c7a5af8de3df2cc4ce';
const loggedModelUrl = `/ajax-api/2.0/mlflow/logged-models/${loggedModelId}`;

describe('ModelVersionSourceModelLink', () => {
  const server = setupServer();

  const renderComponent = () =>
    renderWithIntl(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter>
          <ModelVersionSourceModelLink loggedModelId={loggedModelId} />
        </MemoryRouter>
      </QueryClientProvider>,
    );

  test('shows a skeleton while loading, then links to the logged model page using its name', async () => {
    server.use(
      rest.get(loggedModelUrl, (req, res, ctx) =>
        res(
          ctx.delay(100),
          ctx.json({ model: { info: { model_id: loggedModelId, experiment_id: '123', name: 'iris_model' } } }),
        ),
      ),
    );
    renderComponent();

    expect(screen.getByTestId('source-model-loading')).toBeInTheDocument();
    expect(screen.queryByTestId('source-model-id')).not.toBeInTheDocument();
    expect(screen.queryByTestId('source-model-link')).not.toBeInTheDocument();

    const link = await screen.findByTestId('source-model-link');
    expect(link).toHaveTextContent('iris_model');
    expect(link).toHaveAttribute(
      'href',
      expect.stringContaining(Routes.getExperimentLoggedModelDetailsPageRoute('123', loggedModelId)),
    );
    expect(screen.queryByTestId('source-model-loading')).not.toBeInTheDocument();
  });

  test('uses the model ID as the link text when the logged model has an empty name', async () => {
    server.use(
      rest.get(loggedModelUrl, (req, res, ctx) =>
        res(ctx.json({ model: { info: { model_id: loggedModelId, experiment_id: '123', name: '' } } })),
      ),
    );
    renderComponent();

    expect(await screen.findByTestId('source-model-link')).toHaveTextContent(loggedModelId);
  });

  test('falls back to the plain model ID when the logged model cannot be fetched', async () => {
    server.use(
      rest.get(loggedModelUrl, (req, res, ctx) =>
        res(ctx.status(404), ctx.json({ error_code: 'RESOURCE_DOES_NOT_EXIST' })),
      ),
    );
    renderComponent();

    expect(await screen.findByTestId('source-model-id')).toHaveTextContent(loggedModelId);
    expect(screen.queryByTestId('source-model-link')).not.toBeInTheDocument();
  });

  test('falls back to the plain model ID when the logged model has no experiment ID', async () => {
    server.use(
      rest.get(loggedModelUrl, (req, res, ctx) =>
        res(ctx.json({ model: { info: { model_id: loggedModelId, name: 'iris_model' } } })),
      ),
    );
    renderComponent();

    expect(await screen.findByTestId('source-model-id')).toHaveTextContent(loggedModelId);
    expect(screen.queryByTestId('source-model-link')).not.toBeInTheDocument();
  });
});

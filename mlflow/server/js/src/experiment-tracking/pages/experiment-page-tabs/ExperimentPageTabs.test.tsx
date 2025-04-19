import { DesignSystemProvider } from '@databricks/design-system';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { graphql, rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { setupServer } from '../../../common/utils/setup-msw';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { NOTE_CONTENT_TAG } from '../../utils/NoteUtils';
import ExperimentPageTabs from './ExperimentPageTabs';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  isExperimentLoggedModelsUIEnabled: jest.fn(() => true),
}));

jest.mock('../experiment-logged-models/ExperimentLoggedModelListPage', () => ({
  // mock default export
  __esModule: true,
  default: () => <div>ExperimentLoggedModelListPage</div>,
}));

describe('ExperimentLoggedModelListPage', () => {
  const { history } = setupTestRouter();
  const createTestExperiment = (id = 'test-experiment', name = 'Test experiment name') => {
    return {
      __typename: 'MlflowExperiment',
      artifactLocation: null,
      name,
      creationTime: null,
      experimentId: id,
      lastUpdateTime: null,
      lifecycleStage: 'active',
      tags: [],
    };
  };

  const createTestExperimentResponse = (experiment: any) => ({
    mlflowGetExperiment: {
      __typename: 'MlflowGetExperimentResponse',
      apiError: null,
      experiment,
    },
  });

  const server = setupServer(
    graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
      res(ctx.data(createTestExperimentResponse(createTestExperiment()))),
    ),
  );

  const renderTestComponent = () => {
    return render(
      <TestApolloProvider disableCache>
        <MockedReduxStoreProvider state={{ entities: { experimentTagsByExperimentId: {}, experimentsById: {} } }}>
          <IntlProvider locale="en">
            <DesignSystemProvider>
              <TestRouter
                routes={[testRoute(<ExperimentPageTabs />, '/experiments/:experimentId/:tabName')]}
                history={history}
                initialEntries={['/experiments/test-experiment/models']}
              />
            </DesignSystemProvider>
          </IntlProvider>
        </MockedReduxStoreProvider>
      </TestApolloProvider>,
    );
  };

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(() => {
    server.resetHandlers();
  });

  test('should display experiment title when fetched', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText('Test experiment name')).toBeInTheDocument();
    });
  });

  test('should show error when experiment is missing in API response', async () => {
    server.resetHandlers(
      graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
        res(
          ctx.data({
            mlflowGetExperiment: { experiment: null, apiError: { message: 'The requested resource was not found.' } },
          }),
        ),
      ),
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', (req, res, ctx) => res(ctx.json({ models: [] }))),
    );
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText(/The requested resource was not found./)).toBeInTheDocument();
    });
  });

  test('should edit experiment description', async () => {
    const setTagApiSpy = jest.fn();
    server.use(
      rest.post('/ajax-api/2.0/mlflow/experiments/set-experiment-tag', (req, res, ctx) => {
        setTagApiSpy(req.body);
        return res(ctx.json({}));
      }),
    );
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText('Test experiment name')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: 'Add Description' }));

    await waitFor(() => {
      expect(screen.getByTestId('text-area')).toBeInTheDocument();
    });

    await userEvent.type(screen.getByTestId('text-area'), 'Test description');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(setTagApiSpy).toHaveBeenCalledWith({
      key: NOTE_CONTENT_TAG,
      value: 'Test description',
      experiment_id: 'test-experiment',
    });
  });
});

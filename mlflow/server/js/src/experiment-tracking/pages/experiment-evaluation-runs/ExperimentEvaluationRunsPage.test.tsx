import { jest, describe, beforeAll, beforeEach, test, expect } from '@jest/globals';
import type { DefaultBodyType, PathParams, ResponseResolver, RestRequest, RestContext } from 'msw';
import { rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import ExperimentEvaluationRunsPage from './ExperimentEvaluationRunsPage';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

jest.mock('../../hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(() => ({})),
}));

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000); // higher timeout for heavier table testing

const createMockRun = ({ index }: { index: number }) => ({
  data: {
    params: [],
    tags: [],
    metrics: [],
  },
  info: {
    artifact_uri: '',
    end_time: 0,
    experiment_id: 'exp-1',
    lifecycle_stage: '',
    run_uuid: `run-${index}`,
    run_name: `Test Run ${index}`,
    start_time: 0,
    status: 'FINISHED',
  },
  inputs: {
    dataset_inputs: [],
    model_inputs: [],
  },
  outputs: {
    model_outputs: [],
  },
});
const createMockResponse = ({ pageToken, pageSize }: { pageToken?: string; pageSize: number }) => {
  const allResults = 300;
  const page = Number(pageToken) || 1;
  const allRuns = Array.from({ length: allResults }, (_, i) => createMockRun({ index: i }));
  const next_page_token = page * pageSize < allResults ? (page + 1).toString() : null;
  return {
    runs: allRuns.slice((page - 1) * pageSize, page * pageSize),
    next_page_token,
  };
};

describe('ExperimentEvaluationRunsPage', () => {
  const { history } = setupTestRouter();

  const searchRequestHandler: ResponseResolver<
    RestRequest<DefaultBodyType, PathParams>,
    RestContext,
    DefaultBodyType
  > = (req, res, ctx) => {
    return res(
      ctx.json(
        createMockResponse({
          pageToken: (req.body as any)['page_token'] as string,
          pageSize: Number((req.body as any)['max_results'] as string),
        }),
      ),
    );
  };
  const server = setupServer(
    // prettier-ignore
    rest.post('ajax-api/2.0/mlflow/runs/search', searchRequestHandler),
  );

  const renderTestComponent = () => {
    const queryClient = new QueryClient();
    return render(
      <TestApolloProvider disableCache>
        <QueryClientProvider client={queryClient}>
          <MockedReduxStoreProvider state={{ entities: { colorByRunUuid: {} } }}>
            <IntlProvider locale="en">
              <DesignSystemProvider>
                <TestRouter
                  routes={[testRoute(<ExperimentEvaluationRunsPage />, '/experiments/:experimentId/evaluation-runs')]}
                  history={history}
                  initialEntries={['/experiments/exp-1/evaluation-runs']}
                />
              </DesignSystemProvider>
            </IntlProvider>
          </MockedReduxStoreProvider>
        </QueryClientProvider>
      </TestApolloProvider>,
    );
  };

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(async () => {
    server.resetHandlers();
    renderTestComponent();
    const table = await screen.findByRole('table');
    Object.defineProperty(table, 'scrollHeight', {
      configurable: true,
      value: 1000,
    });
    Object.defineProperty(table, 'scrollTop', {
      configurable: true,
      writable: true,
      value: 0,
    });
  });

  test('should display runs title when fetched', async () => {
    await waitFor(() => {
      // Make sure first and last items are displayed.
      expect(screen.getByText('Test Run 0')).toBeInTheDocument();
      expect(screen.getByText('Test Run 49')).toBeInTheDocument();
    });
    // Make sure the next page is not yet fetched.
    expect(screen.queryByText('Test Run 50')).not.toBeInTheDocument();
  });

  test('should load next page when scroll is at the bottom', async () => {
    fireEvent.scroll(screen.getByRole('table'), { target: { scrollTop: 1000 } });
    await waitFor(() => {
      // Make sure first and last items are displayed.
      expect(screen.getByText('Test Run 50')).toBeInTheDocument();
      expect(screen.getByText('Test Run 99')).toBeInTheDocument();
    });
  });
});

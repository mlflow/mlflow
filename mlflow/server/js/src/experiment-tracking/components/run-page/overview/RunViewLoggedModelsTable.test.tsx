import { IntlProvider } from 'react-intl';
import { render, waitFor } from '../../../../common/utils/TestUtils.react18';
import { LoggedModelStatusProtoEnum, type RunInfoEntity } from '../../../types';
import type { UseGetRunQueryResponseInputs, UseGetRunQueryResponseOutputs } from '../hooks/useGetRunQuery';
import { RunViewLoggedModelsTable } from './RunViewLoggedModelsTable';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { TestApolloProvider } from '../../../../common/utils/TestApolloProvider';
import { DesignSystemProvider } from '@databricks/design-system';
import { TestRouter, testRoute } from '../../../../common/utils/RoutingTestUtils';

jest.setTimeout(90000); // High timeout because of testing heavy data table

describe('RunViewLoggedModelsTable', () => {
  const server = setupServer(
    rest.get('/ajax-api/2.0/mlflow/logged-models/:modelId', (req, res, ctx) => {
      const modelId = req.params['modelId'].toString();
      return res(
        ctx.json({
          model: {
            info: {
              artifact_uri: `dbfs:/databricks/mlflow/${modelId}`,
              creation_timestamp_ms: 1728322600000,
              last_updated_timestamp_ms: 1728322600000,
              source_run_id: 'run-id-1',
              creator_id: 'test@test.com',
              experiment_id: 'test-experiment',
              model_id: modelId,
              model_type: 'Agent',
              name: `${modelId}-name`,
              tags: [],
              status_message: 'Ready',
              status: LoggedModelStatusProtoEnum.LOGGED_MODEL_READY,
              registrations: [],
            },
            data: {},
          },
        }),
      );
    }),
  );
  const inputs: UseGetRunQueryResponseInputs = {
    __typename: 'MlflowRunInputs',
    datasetInputs: null,
    modelInputs: [
      { __typename: 'MlflowModelInput', modelId: 'input-model-1' },
      { __typename: 'MlflowModelInput', modelId: 'input-model-2' },
    ],
  };

  const outputs: UseGetRunQueryResponseOutputs = {
    __typename: 'MlflowRunOutputs',
    modelOutputs: [
      { __typename: 'MlflowModelOutput', modelId: 'output-model-1', step: '0' },
      { __typename: 'MlflowModelOutput', modelId: 'output-model-2', step: '0' },
    ],
  };
  const runInfo: RunInfoEntity = {
    runUuid: 'run-id',
    experimentId: 'experiment-id',
    startTime: 0,
    endTime: 1,
    artifactUri: '',
    lifecycleStage: 'active',
    status: 'FINISHED',
    runName: 'run-name',
  };

  const renderTestComponent = () => {
    const queryClient = new QueryClient();
    return render(<RunViewLoggedModelsTable inputs={inputs} outputs={outputs} runInfo={runInfo} />, {
      wrapper: ({ children }) => (
        <TestApolloProvider>
          <QueryClientProvider client={queryClient}>
            <DesignSystemProvider>
              <IntlProvider locale="en">
                <TestRouter routes={[testRoute(<>{children}</>)]} />
              </IntlProvider>
            </DesignSystemProvider>
          </QueryClientProvider>
        </TestApolloProvider>
      ),
    });
  };
  it('renders a table with logged models', async () => {
    const { getByText, getByRole, queryByText, getAllByRole } = renderTestComponent();

    await waitFor(() => {
      expect(getByText(/Logged models \(\d+\)/)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(queryByText('Models loading')).not.toBeInTheDocument();
    });

    expect(getByRole('grid')).toBeInTheDocument();

    // Wait for the first cell to appear
    await waitFor(() => {
      expect(getByRole('gridcell', { name: /input-model-1-name/ })).toBeInTheDocument();
    });

    // Assert that all cells are rendered
    expect(getByRole('gridcell', { name: /input-model-2-name/ })).toBeInTheDocument();
    expect(getByRole('gridcell', { name: /output-model-1-name/ })).toBeInTheDocument();
    expect(getByRole('gridcell', { name: /output-model-2-name/ })).toBeInTheDocument();

    expect(getAllByRole('gridcell', { name: 'Input' })).toHaveLength(2);
    expect(getAllByRole('gridcell', { name: 'Output' })).toHaveLength(2);
  });

  it('renders error message when request fails', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});

    server.use(
      rest.get('/ajax-api/2.0/mlflow/logged-models/:modelId', (req, res, ctx) => {
        return res(ctx.status(500), ctx.json({ message: 'Something went wrong.' }));
      }),
    );

    const { getByText, queryByText } = renderTestComponent();

    await waitFor(() => {
      expect(getByText(/Logged models \(\d+\)/)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(queryByText('Models loading')).not.toBeInTheDocument();
    });

    expect(getByText('Something went wrong.')).toBeInTheDocument();

    jest.restoreAllMocks();
  });
});

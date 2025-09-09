import { IntlProvider } from 'react-intl';
import { render, waitFor } from '../../../../common/utils/TestUtils.react18';
import type { LoggedModelProto } from '../../../types';
import { LoggedModelStatusProtoEnum, type RunInfoEntity } from '../../../types';
import type { UseGetRunQueryResponseInputs, UseGetRunQueryResponseOutputs } from '../hooks/useGetRunQuery';
import { RunViewLoggedModelsTable } from './RunViewLoggedModelsTable';
import { TestApolloProvider } from '../../../../common/utils/TestApolloProvider';
import { DesignSystemProvider } from '@databricks/design-system';
import { TestRouter, testRoute } from '../../../../common/utils/RoutingTestUtils';
import type { ComponentProps } from 'react';
import { QueryClient, QueryClientProvider } from '../../../../common/utils/reactQueryHooks';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(90000); // High timeout because of testing heavy data table

describe('RunViewLoggedModelsTable', () => {
  const testLoggedModels: LoggedModelProto[] = [
    'input-model-1',
    'input-model-2',
    'output-model-1',
    'output-model-2',
  ].map((modelId) => ({
    info: {
      artifact_uri: `dbfs:/databricks/mlflow/${modelId}`,
      creation_timestamp_ms: 1728322600000,
      last_updated_timestamp_ms: 1728322600000,
      source_run_id: 'run-id-1',
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
  }));

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
      { __typename: 'MlflowModelOutput', modelId: 'output-model-1', step: '2' },
      { __typename: 'MlflowModelOutput', modelId: 'output-model-2', step: '7' },
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

  const renderTestComponent = (props: Partial<ComponentProps<typeof RunViewLoggedModelsTable>> = {}) => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    return render(
      <RunViewLoggedModelsTable
        inputs={inputs}
        outputs={outputs}
        runInfo={runInfo}
        loggedModelsV3={testLoggedModels}
        isLoadingLoggedModels={false}
        {...props}
      />,
      {
        wrapper: ({ children }) => (
          <QueryClientProvider client={queryClient}>
            <TestApolloProvider>
              <DesignSystemProvider>
                <IntlProvider locale="en">
                  <TestRouter routes={[testRoute(<>{children}</>)]} />
                </IntlProvider>
              </DesignSystemProvider>
            </TestApolloProvider>
          </QueryClientProvider>
        ),
      },
    );
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

  it('renders corresponding steps for logged models', async () => {
    const { getByRole } = renderTestComponent();

    // Wait for the first cell to appear
    await waitFor(() => {
      expect(getByRole('gridcell', { name: /output-model-1-name/ })).toBeInTheDocument();
    });

    const [outputModelOneRow, outputModelTwoRow] = [/output-model-1-name/, /output-model-2-name/].map(
      (cellContent) => getByRole('gridcell', { name: cellContent }).closest('[role="row"]') as HTMLElement,
    );

    // Due to lack of accessibility labels in ag-grid, we are using column IDs
    const stepColId = getByRole('columnheader', { name: 'Step' }).getAttribute('col-id');

    expect(outputModelOneRow.querySelector(`[col-id="${stepColId}"]`)).toHaveTextContent('2');
    expect(outputModelTwoRow.querySelector(`[col-id="${stepColId}"]`)).toHaveTextContent('7');
  });

  it('renders error message when  provided', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});

    const { getByText, queryByText } = renderTestComponent({
      loggedModelsError: new Error('Something went wrong.'),
    });

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

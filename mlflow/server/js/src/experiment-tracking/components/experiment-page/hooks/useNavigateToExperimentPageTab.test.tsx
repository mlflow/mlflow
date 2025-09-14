import { render, renderHook, screen, waitFor } from '@testing-library/react';
import { useNavigateToExperimentPageTab } from './useNavigateToExperimentPageTab';
import { setupTestRouter, testRoute, TestRouter } from '../../../../common/utils/RoutingTestUtils';
import { createMLflowRoutePath, useParams } from '../../../../common/utils/RoutingUtils';
import { TestApolloProvider } from '../../../../common/utils/TestApolloProvider';
import { setupServer } from '../../../../common/utils/setup-msw';
import { graphql } from 'msw';
import type { MlflowGetExperimentQuery } from '../../../../graphql/__generated__/graphql';
import { ExperimentKind } from '../../../constants';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(60000); // Larger timeout for integration testing

describe('useNavigateToExperimentPageTab', () => {
  const server = setupServer();

  beforeEach(() => {
    server.resetHandlers();
  });

  const { history } = setupTestRouter();

  const mockResponseWithExperimentKind = (experimentKind: ExperimentKind) => {
    server.use(
      graphql.query<MlflowGetExperimentQuery>('MlflowGetExperimentQuery', (req, res, ctx) => {
        return res(
          ctx.data({
            mlflowGetExperiment: {
              __typename: 'MlflowGetExperimentResponse',
              experiment: {
                tags: [
                  {
                    __typename: 'MlflowExperimentTag',
                    key: 'mlflow.experimentKind',
                    value: experimentKind,
                  },
                ],
              },
            } as any,
          }),
        );
      }),
    );
  };

  const renderTestHook = (initialRoute: string, enabled = true) => {
    const TestExperimentPage = () => {
      const { isEnabled } = useNavigateToExperimentPageTab({
        enabled,
        experimentId: '123',
      });

      if (isEnabled) {
        return null;
      }

      return <span>this is default entry page</span>;
    };
    const TestExperimentTabsPage = () => {
      // /experiments/:experimentId/:tabName
      const { tabName } = useParams();
      return <span>experiment page displaying {tabName} tab</span>;
    };
    return render(
      <TestRouter
        history={history}
        routes={[
          testRoute(<TestExperimentPage />, createMLflowRoutePath('/experiments/:experimentId')),
          testRoute(<TestExperimentTabsPage />, createMLflowRoutePath('/experiments/:experimentId/:tabName')),
        ]}
        initialEntries={[initialRoute]}
      />,
      { wrapper: ({ children }) => <TestApolloProvider disableCache>{children}</TestApolloProvider> },
    );
  };
  test('should not redirect if the hook is disabled', async () => {
    renderTestHook(createMLflowRoutePath('/experiments/123'), false);

    await waitFor(() => {
      expect(screen.getByText('this is default entry page')).toBeInTheDocument();
    });
  });

  test('should redirect to the traces tab on GenAI experiment kind', async () => {
    mockResponseWithExperimentKind(ExperimentKind.GENAI_DEVELOPMENT);

    renderTestHook(createMLflowRoutePath('/experiments/123'));

    expect(await screen.findByText('experiment page displaying traces tab')).toBeInTheDocument();
  });

  test('should redirect to the traces tab on custom development experiment kind', async () => {
    mockResponseWithExperimentKind(ExperimentKind.CUSTOM_MODEL_DEVELOPMENT);

    renderTestHook(createMLflowRoutePath('/experiments/123'));

    expect(await screen.findByText('experiment page displaying runs tab')).toBeInTheDocument();
  });
});

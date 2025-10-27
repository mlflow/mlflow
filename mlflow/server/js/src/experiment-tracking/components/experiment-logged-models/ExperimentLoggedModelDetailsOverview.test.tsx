import type { ComponentProps } from 'react';
import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { ExperimentLoggedModelDetailsOverview } from './ExperimentLoggedModelDetailsOverview';
import { ExperimentKind } from '../../constants';
import { IntlProvider } from 'react-intl';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MockedProvider } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';

jest.mock('../../hooks/logged-models/useRelatedRunsDataForLoggedModels', () => ({
  useRelatedRunsDataForLoggedModels: () => ({
    data: [],
    loading: false,
    error: null,
  }),
}));

describe('ExperimentLoggedModelDetailsOverview', () => {
  const renderTestComponent = (props: Partial<ComponentProps<typeof ExperimentLoggedModelDetailsOverview>> = {}) => {
    render(
      <ExperimentLoggedModelDetailsOverview
        onDataUpdated={jest.fn()}
        loggedModel={{ data: {}, info: { name: 'TestModel', model_id: 'm-123456' } }}
        {...props}
      />,
      {
        wrapper: ({ children }) => (
          <TestRouter
            routes={[
              testRoute(
                <IntlProvider locale="en">
                  <MockedProvider>
                    <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
                  </MockedProvider>
                </IntlProvider>,
                '*',
              ),
            ]}
          />
        ),
      },
    );
  };

  test('Logged model overview contains linked prompts when experiment is of GenAI type', async () => {
    renderTestComponent({ experimentKind: ExperimentKind.GENAI_DEVELOPMENT });

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Prompts' })).toBeInTheDocument();
    });
  });

  test('Logged model overview skips linked prompts box when experiment is not of GenAI type', async () => {
    renderTestComponent({ experimentKind: ExperimentKind.CUSTOM_MODEL_DEVELOPMENT });

    await waitFor(() => {
      expect(screen.getByText('m-123456')).toBeInTheDocument();
    });

    expect(screen.queryByRole('heading', { name: 'Prompts' })).not.toBeInTheDocument();
  });
});

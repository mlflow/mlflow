import { useState, type ReactNode } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { MockedProvider, type MockedResponse } from '../../../common/utils/graphQLHooks';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import type { GetRun } from '../../../graphql/__generated__/graphql';
import { GET_RUN_QUERY } from '../../components/run-page/hooks/useGetRunQuery';
import { createComparisonRun } from './evalRunComparisonTestUtils';

export const createComparisonRunMock = (runUuid: string): MockedResponse<GetRun> => ({
  request: { query: GET_RUN_QUERY, variables: { data: { runId: runUuid } } },
  result: {
    data: {
      mlflowGetRun: {
        __typename: 'MlflowGetRunResponse',
        apiError: null,
        run: createComparisonRun(runUuid),
      },
    },
  },
});

/** Providers the comparison components need in tests: intl, design system, Apollo, Redux, router, query client. */
export const EvalRunComparisonTestProviders = ({
  children,
  client,
  apolloMocks = ['baseline', 'candidate', 'third'].map(createComparisonRunMock),
}: {
  children: ReactNode;
  client: QueryClient;
  apolloMocks?: MockedResponse[];
}) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>
      <MockedProvider mocks={apolloMocks}>
        <MockedReduxStoreProvider state={{ entities: { colorByRunUuid: {} } }}>
          <MemoryRouter>
            <QueryClientProvider client={client}>{children}</QueryClientProvider>
          </MemoryRouter>
        </MockedReduxStoreProvider>
      </MockedProvider>
    </DesignSystemProvider>
  </IntlProvider>
);

/** `EvalRunComparisonTestProviders` with its own query client, for a render's `wrapper` option. */
export const EvalRunComparisonTestWrapper = ({ children }: { children?: ReactNode }) => {
  const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }));
  return <EvalRunComparisonTestProviders client={client}>{children}</EvalRunComparisonTestProviders>;
};

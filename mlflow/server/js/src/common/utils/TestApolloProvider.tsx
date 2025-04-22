import { ApolloProvider } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { createApolloClient } from '../../graphql/client';
import { useMemo } from 'react';

export function TestApolloProvider({
  children,
  disableCache,
}: React.PropsWithChildren<{
  disableCache?: boolean;
}>) {
  const client = useMemo(() => {
    const apolloClient = createApolloClient();
    if (disableCache) {
      apolloClient.defaultOptions = {
        watchQuery: {
          fetchPolicy: 'no-cache',
        },
        query: {
          fetchPolicy: 'no-cache',
        },
      };
    }
    return apolloClient;
  }, [disableCache]);

  return <ApolloProvider client={client}>{children}</ApolloProvider>;
}

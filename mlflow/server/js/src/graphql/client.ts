import { ApolloClient, ApolloLink, InMemoryCache, Operation, createHttpLink } from '@apollo/client';
import { RetryLink } from '@apollo/client/link/retry';

function containsMutation(op: Operation): boolean {
  const definitions = (op.query && op.query.definitions) || [];
  return Boolean(definitions.find((d) => d.kind === 'OperationDefinition' && d.operation === 'mutation'));
}

const backgroundLinkTimeoutMs = 10000;

const possibleTypes: Record<string, string[]> = {};

const graphqlFetch = async (uri: any, options: any): Promise<Response> => {
  const headers = new Headers({
    ...options.headers,
  });

  return fetch(uri, { ...options, headers }).then((res) => res);
};

const apolloCache = new InMemoryCache({
  possibleTypes,
  typePolicies: {
    Query: {
      fields: {},
    },
  },
});

export function createApolloClient() {
  const httpLink = createHttpLink({
    uri: '/graphql',
    credentials: 'same-origin',
    fetch: graphqlFetch,
  });

  // Copied from redash -- I guess the idea is to retry if the request isn't a mutation?
  const retryLink = new RetryLink({
    attempts: { retryIf: (_, op) => !containsMutation(op) },
  });

  // eslint-disable-next-line prefer-const
  let combinedLinks = ApolloLink.from([
    // This link retries queries that fail due to network errors
    retryLink,
    httpLink,
  ]);

  return new ApolloClient({
    link: combinedLinks,
    cache: apolloCache,
  });
}

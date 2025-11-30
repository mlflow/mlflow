import {
  ApolloClient,
  ApolloLink,
  InMemoryCache,
  type Operation,
  createHttpLink,
} from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import {
  // prettier-ignore
  RetryLink,
  DefaultHeadersLink as OssDefaultHeadersLink,
} from '@mlflow/mlflow/src/common/utils/graphQLHooks';

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

  // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
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
  const uri = 'graphql';
  const httpLink = createHttpLink({
    uri,
    credentials: 'same-origin',
    fetch: graphqlFetch,
  });

  // Copied from redash -- I guess the idea is to retry if the request isn't a mutation?
  const retryLink = new RetryLink({
    attempts: { retryIf: (_, op) => !containsMutation(op) },
  });

  const DefaultHeadersLink = OssDefaultHeadersLink;
  const defaultHeadersLink = new DefaultHeadersLink({
    cookieStr: document.cookie,
  });

  // eslint-disable-next-line prefer-const
  let combinedLinks = ApolloLink.from([
    // This link retries queries that fail due to network errors
    retryLink,
    // This link adds application-specific headers to HTTP requests (e.g., CSRF tokens)
    defaultHeadersLink,
    httpLink,
  ]);

  return new ApolloClient({
    link: combinedLinks,
    cache: apolloCache,
  });
}

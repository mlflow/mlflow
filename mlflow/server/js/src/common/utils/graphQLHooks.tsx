import type { Observable } from '@apollo/client/core';
import { ApolloLink, type Operation, type NextLink, type FetchResult } from '@apollo/client/core';
import { getDefaultHeaders } from './FetchUtils';
export * from '@apollo/client';
export * from '@apollo/client/link/retry';
export * from '@apollo/client/testing';

export class DefaultHeadersLink extends ApolloLink {
  private cookieStr: string;

  constructor({ cookieStr }: { cookieStr: string }) {
    super();
    this.cookieStr = cookieStr;
  }

  request(operation: Operation, forward: NextLink): Observable<FetchResult> {
    operation.setContext(({ headers = {} }) => ({
      headers: {
        ...headers,
        ...getDefaultHeaders(this.cookieStr),
      },
    }));

    return forward(operation);
  }
}

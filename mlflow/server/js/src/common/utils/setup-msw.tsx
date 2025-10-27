import type { RequestHandler } from 'msw';
import { setupServer as setupMsw } from 'msw/node';

export function setupServer(...handlers: RequestHandler[]) {
  const server = setupMsw(...handlers);

  beforeAll(() => {
    server.listen({
      onUnhandledRequest: 'warn',
    });
  });

  beforeEach(() => {
    // In order to make graphql work with msw, we need to restore the fetch function so that it
    // automatically throw an error
    // This is only relevant for when the graphql queries are called.
    jest.mocked(global.fetch).mockRestore();
  });

  afterEach(() => server.resetHandlers());
  afterAll(() => server.close());

  return server;
}

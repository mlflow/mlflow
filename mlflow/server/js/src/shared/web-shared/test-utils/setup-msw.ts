import { afterAll, afterEach, beforeAll, beforeEach } from '@jest/globals';
import { waitFor } from '@testing-library/react';
import type { MockedRequest, RequestHandler } from 'msw';
// eslint-disable-next-line @databricks/no-restricted-imports-regexp -- this is the test-utils wrapper; the msw/node boundary lives here by design
import type { SetupServer as BaseSetupServer } from 'msw/node';
import { matchRequestUrl } from 'msw';
// eslint-disable-next-line @databricks/no-restricted-imports-regexp -- this is the test-utils wrapper; the msw/node boundary lives here by design
import { setupServer as setupMsw } from 'msw/node';

export type SetupServer = Omit<BaseSetupServer, 'listen' | 'close'>;

export type WaitForRequest = (
  method: 'get' | 'post' | 'put' | 'delete' | 'patch',
  url: string | RegExp,
) => Promise<MockedRequest>;

export type WaitForPendingRequests = () => Promise<void>;

/**
 * Wraps `setupServer` from `msw/node` with automatic Jest lifecycle integration.
 *
 * - Starts the server in `beforeAll`, resets handlers in `afterEach`, closes in `afterAll`.
 * - Throws when requests are still in-flight when a test ends — a common cause of test flakiness.
 * - Throws on requests that don't match any registered handler.
 * - Returns `waitForRequest` to assert on specific HTTP calls made during a test.
 * - Hides `listen` / `close` from the returned `server` so callers can't bypass the
 *   auto-registered lifecycle and cause double-listen flakiness.
 * - Guards `resetHandlers` at runtime: a no-arg call (redundant with the auto-managed
 *   `afterEach`) throws, while mid-test handler swaps `resetHandlers(h1, h2)` are allowed.
 *
 * Import from `@databricks/web-shared/test-utils`. Do NOT import `setupServer` directly
 * from `msw/node`.
 *
 * @example
 * describe('MyComponent', () => {
 *   const { server, waitForRequest } = setupServer();
 *
 *   test('submits form', async () => {
 *     server.use(rest.post('/api/submit', (req, res, ctx) => res(ctx.status(200))));
 *     // ... render and act ...
 *     const req = await waitForRequest('post', '/api/submit');
 *     expect(req.body).toEqual({ name: 'test' });
 *   });
 * });
 */
export function setupServer(...requestHandlers: RequestHandler[]): {
  server: SetupServer;
  waitForRequest: WaitForRequest;
  waitForPendingRequests: WaitForPendingRequests;
} {
  const server = setupMsw(...requestHandlers);
  const pendingRequests = new Map<string, MockedRequest>();
  const matchedRequests = new Map<string, MockedRequest>();

  server.events.on('request:start', (req) => {
    pendingRequests.set(req.id, req);
  });

  server.events.on('request:end', (req) => {
    pendingRequests.delete(req.id);
  });

  // When a handler throws (including res.networkError() and failing test assertions),
  // request:end never fires. Clean up manually so afterEach doesn't false-positive.
  server.events.on('unhandledException', (_error, req) => {
    pendingRequests.delete(req.id);
  });

  server.events.on('request:match', (req) => {
    matchedRequests.set(req.id, req);
  });

  // Capture internal methods before overriding, so lifecycle hooks can still call them.
  const internalListen = server.listen.bind(server);
  const internalClose = server.close.bind(server);
  const internalResetHandlers = server.resetHandlers.bind(server);

  // Guard against bypassing the auto-registered lifecycle: calling listen/close manually
  // causes inconsistent request interception and double-listen flakiness. No-arg
  // resetHandlers is redundant with the auto-managed afterEach — allow only the
  // mid-test handler-swap form `resetHandlers(h1, h2, ...)`.
  server.listen = () => {
    throw new Error(
      'Do not call server.listen() manually — setupServer() registers this in beforeAll automatically. Use server.use() to add request handlers per test.',
    );
  };
  server.close = () => {
    throw new Error('Do not call server.close() manually — setupServer() manages teardown in afterAll.');
  };
  server.resetHandlers = ((...handlers: RequestHandler[]) => {
    if (handlers.length === 0) {
      throw new Error(
        'Do not call server.resetHandlers() with no args — setupServer() resets in afterEach automatically. To swap handlers mid-test, pass the new handlers as args.',
      );
    }
    internalResetHandlers(...handlers);
  }) as BaseSetupServer['resetHandlers'];

  beforeAll(() => {
    internalListen({
      onUnhandledRequest: (req, { error }) => {
        // MSW 0.39.x bug: unhandled requests skip request:end, leaving phantom pending
        // entries in the map. Remove it manually before throwing so tracking stays accurate.
        pendingRequests.delete(req.id);
        error();
      },
    });
  });

  beforeEach(() => {
    pendingRequests.clear();
    matchedRequests.clear();
  });

  afterEach(() => {
    if (pendingRequests.size > 0) {
      const message = [
        'Pending network requests at test end (typically causes flaky behavior):',
        ...Array.from(pendingRequests.values()).map((r) => `  ${r.method} ${r.url.href}`),
      ].join('\n');
      throw new Error(message);
    }
    internalResetHandlers();
  });

  afterAll(() => {
    internalClose();
  });

  const waitForRequest: WaitForRequest = (method, url) => {
    const notFound = new Error(`Request ${method.toUpperCase()} ${String(url)} was not made`);
    return waitFor(() => {
      const match = Array.from(matchedRequests.values()).find(
        (req) => req.method.toLowerCase() === method && matchRequestUrl(req.url, url).matches,
      );
      if (!match) throw notFound;
      matchedRequests.delete(match.id);
      return match;
    });
  };

  const waitForPendingRequests: WaitForPendingRequests = () =>
    waitFor(() => {
      if (pendingRequests.size > 0) {
        throw new Error(`Still ${pendingRequests.size} pending request(s)`);
      }
    });

  return { server, waitForRequest, waitForPendingRequests };
}

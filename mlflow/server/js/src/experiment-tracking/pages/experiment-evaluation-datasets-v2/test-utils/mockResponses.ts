interface MockResponseInit {
  status?: number;
  ok?: boolean;
  statusText?: string;
}

// Test helper that returns a Fetch `Response`-shaped object for code paths that mock
// `workspaceFetch`. The underlying production code only reads `ok`, `status`, `statusText`,
// and `json()`, so a full polyfill of the Response interface would be wasted ceremony.
// The unsafe cast is contained here once — callers stay fully typed.
export const mockJsonResponse = <T>(body: T, init: MockResponseInit = {}): Response => {
  const status = init.status ?? 200;
  const ok = init.ok ?? (status >= 200 && status < 300);
  const statusText = init.statusText ?? '';
  return { ok, status, statusText, json: async () => body } as unknown as Response;
};

export const mockEmptyResponse = (init: MockResponseInit = {}): Response => mockJsonResponse({}, init);

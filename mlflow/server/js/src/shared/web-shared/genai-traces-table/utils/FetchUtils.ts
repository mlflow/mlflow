import cookie from 'cookie';

// eslint-disable-next-line no-restricted-globals
export const fetchFn = fetch; // use global fetch for oss

const WORKSPACE_STORAGE_KEY = 'mlflow.activeWorkspace';

/**
 * Get the currently active workspace from localStorage.
 * This is a minimal implementation for the shared library that cannot depend on main mlflow code.
 */
const getActiveWorkspace = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    return window.localStorage.getItem(WORKSPACE_STORAGE_KEY);
  } catch {
    return null;
  }
};

/**
 * Parse cookies to extract request headers.
 * Minimal implementation for shared library.
 */
export const getDefaultHeadersFromCookies = (cookieStr: string) => {
  const headerCookiePrefix = 'mlflow-request-header-';
  const parsedCookie = cookie.parse(cookieStr);
  if (!parsedCookie || Object.keys(parsedCookie).length === 0) {
    return {};
  }
  return Object.keys(parsedCookie)
    .filter((cookieName) => cookieName.startsWith(headerCookiePrefix))
    .reduce(
      (acc, cookieName) => ({
        ...acc,
        [cookieName.substring(headerCookiePrefix.length)]: parsedCookie[cookieName],
      }),
      {},
    );
};

/**
 * Get default headers including workspace header if active.
 * Minimal implementation for shared library.
 */
export const getDefaultHeaders = (cookieStr: string) => {
  const cookieHeaders = getDefaultHeadersFromCookies(cookieStr);
  const workspace = getActiveWorkspace();

  return {
    ...cookieHeaders,
    ...(workspace ? { 'X-MLFLOW-WORKSPACE': workspace } : {}),
  };
};

/**
 * Convert relative URL to absolute if needed.
 * Minimal implementation for shared library.
 */
export const getAjaxUrl = (relativeUrl: string) => {
  if (
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] === 'true' &&
    typeof relativeUrl === 'string' &&
    !relativeUrl.startsWith('/')
  ) {
    return '/' + relativeUrl;
  }
  return relativeUrl;
};

/**
 * Helper method to make a request to the backend with workspace support.
 * Minimal implementation for shared library.
 */
export const fetchAPI = async (url: string, options: Omit<RequestInit, 'body'> & { body?: any } = {}) => {
  const { method, headers, body, ...restOptions } = options;

  let cookieString = '';
  if (typeof document !== 'undefined' && typeof document.cookie === 'string') {
    cookieString = document.cookie || '';
  }

  const serializeBody = (payload: any) => {
    if (payload === undefined) {
      return undefined;
    }
    return typeof payload === 'string' || payload instanceof FormData || payload instanceof Blob
      ? payload
      : JSON.stringify(payload);
  };

  const fetchOptions: RequestInit = {
    ...restOptions,
    method: method || 'GET',
    headers: {
      ...getDefaultHeaders(cookieString),
      ...(body ? { 'Content-Type': 'application/json' } : {}),
      ...headers,
    },
    ...(body && { body: serializeBody(body) }),
  };

  // eslint-disable-next-line no-restricted-globals
  const response = await fetch(url, fetchOptions);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json();
};

export const makeRequest = async <T>(path: string, method: 'POST' | 'GET', body?: T, signal?: AbortSignal) => {
  return fetchAPI(path, { method, body, signal });
};

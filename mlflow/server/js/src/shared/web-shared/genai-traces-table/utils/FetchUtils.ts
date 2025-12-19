import { matchPredefinedError } from '../../errors';

// eslint-disable-next-line no-restricted-globals
export const fetchFn = fetch; // use global fetch for oss

export const makeRequest = async <T>(path: string, method: 'POST' | 'GET', body?: T, signal?: AbortSignal) => {
  const options: RequestInit = {
    method,
    signal,
    headers: {
      ...(body ? { 'Content-Type': 'application/json' } : {}),
      ...getDefaultHeaders(document.cookie),
    },
  };

  if (body) {
    options.body = JSON.stringify(body);
  }
  const response = await fetchFn(path, options);

  if (!response.ok) {
    const error = matchPredefinedError(response);
    try {
      const errorMessageFromResponse = await (await response.json()).message;
      if (errorMessageFromResponse) {
        error.message = errorMessageFromResponse;
      }
    } catch {
      // do nothing
    }
    throw error;
  }

  return response.json();
};

export const getAjaxUrl = (relativeUrl: any) => {
  if (process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] === 'true' && !relativeUrl.startsWith('/')) {
    return '/' + relativeUrl;
  }
  return relativeUrl;
};

// Parse cookies from document.cookie
function parseCookies(cookieString = document.cookie) {
  return cookieString.split(';').reduce((cookies: { [key: string]: string }, cookie: string) => {
    const [name, value] = cookie.trim().split('=');
    cookies[name] = decodeURIComponent(value || '');
    return cookies;
  }, {});
}

export const getDefaultHeadersFromCookies = (cookieStr: any) => {
  const headerCookiePrefix = 'mlflow-request-header-';
  const parsedCookie = parseCookies(cookieStr);
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

export const getDefaultHeaders = (cookieStr: any) => {
  const cookieHeaders = getDefaultHeadersFromCookies(cookieStr);
  return {
    ...cookieHeaders,
  };
};

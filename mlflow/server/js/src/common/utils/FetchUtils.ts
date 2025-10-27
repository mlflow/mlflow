/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import cookie from 'cookie';
import JsonBigInt from 'json-bigint';
import yaml from 'js-yaml';
import { pickBy } from 'lodash';
import { ErrorWrapper } from './ErrorWrapper';
import { matchPredefinedError } from '@databricks/web-shared/errors';

export const HTTPMethods = {
  GET: 'GET',
  POST: 'POST',
  PUT: 'PUT',
  PATCH: 'PATCH',
  DELETE: 'DELETE',
};

// HTTP status codes that should be retried. Includes:
// 429 (too many requests), 556 (RCP: workspace not served by shard)
export const HTTPRetryStatuses = [429, 556];

// To enable running behind applications that require specific headers
// to be set during HTTP requests (e.g., CSRF tokens), we support parsing
// a set of cookies with a key prefix of "$appName-request-header-$headerName",
// which will be added as an HTTP header to all requests.
export const getDefaultHeadersFromCookies = (cookieStr: any) => {
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

export const getDefaultHeaders = (cookieStr: any) => {
  const cookieHeaders = getDefaultHeadersFromCookies(cookieStr);
  return {
    ...cookieHeaders,
  };
};

export const getAjaxUrl = (relativeUrl: any) => {
  if (process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] === 'true' && !relativeUrl.startsWith('/')) {
    return '/' + relativeUrl;
  }
  return relativeUrl;
};

// return response json by default, if response is not parsable to json,
// return response text as best effort.
// e.g. model artifact files are in yaml format. Currently it is parsed in a separate action.
// We should remove the redundant action and use the yaml parser defined here
export const parseResponse = ({ resolve, response, parser }: any) => {
  response.text().then((text: any) => {
    try {
      resolve(parser(text));
    } catch {
      resolve(text);
    }
  });
};

export const defaultResponseParser = ({ resolve, response }: any) =>
  parseResponse({ resolve, response, parser: JSON.parse });

export const jsonBigIntResponseParser = ({ resolve, response }: any) =>
  parseResponse({
    resolve,
    response,
    parser: JsonBigInt({ strict: true, storeAsString: true }).parse,
  });

export const yamlResponseParser = ({ resolve, response }: any) =>
  parseResponse({ resolve, response, parser: yaml.safeLoad });

export const defaultError = ({ reject, response, err }: any) => {
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.error('Fetch failed: ', response || err);
  if (response) {
    response.text().then((text: any) => reject(new ErrorWrapper(text, response.status)));
  } else if (err) {
    reject(new ErrorWrapper(err, 500));
  }
};

/**
 * Makes a fetch request.
 * Note this is not intended to be used outside of this file,
 * use `fetchEndpoint` instead.
 */
export const fetchEndpointRaw = ({
  relativeUrl,
  method = HTTPMethods.GET,
  body = undefined,
  headerOptions = {},
  options = {},
  timeoutMs = undefined,
}: any) => {
  const url = getAjaxUrl(relativeUrl);

  // if custom headers has duplicate fields with default Headers,
  // values in the custom headers options will always override.
  const headers = {
    'Content-Type': 'application/json; charset=utf-8',
    ...getDefaultHeaders(document.cookie),
    ...headerOptions,
  };

  const defaultOptions = {
    dataType: 'json',
  };
  // use an abort controller for setting request timeout if defined
  // https://stackoverflow.com/questions/46946380/fetch-api-request-timeout
  const abortController = new AbortController();
  if (timeoutMs) {
    setTimeout(() => abortController.abort(), timeoutMs);
  }

  const fetchOptions = {
    method,
    headers,
    ...(body && { body }),
    ...defaultOptions,
    ...options,
    ...(timeoutMs && { signal: abortController.signal }),
  };
  // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
  return fetch(url, fetchOptions);
};

/**
 * Generic function to retry a given function
 * @param fn: function to retry
 * @param options: additional options
 * - retries: max number of retries
 * - interval: wait interval before the next retry
 * - retryIntervalMultiplier: wait interval multiplier for each additional retry
 * - successCondition: callback with the result of `fn` to determine
 * if `success` callback should be triggered.
 * Defaults to `true`, meaning as long as no exception is thrown,
 * we should trigger `success` callback.
 * - success: callback with the result of `fn` when `successCondition` is true,
 * Defaults to returning the response
 * - errorCondition: callback with the result of `fn` to determine
 * if `error` callback should be triggered.
 * Defaults to `false`, meaning as long as we got a response,
 * we should not trigger `error` callback.
 * - error: callback with the result of `fn` when any of the following case is met
 * 1. errorCondition is true
 * 2. the max number of retries has reached
 * 3. an exception error is thrown while executing `fn`
 * Defaults to throwing an exception
 * @returns {Promise<*|undefined>}
 */
// @ts-expect-error TS(7023): 'retry' implicitly has return type 'any' because i... Remove this comment to see the full error message
export const retry = async (
  fn: any,
  {
    retries = 2,
    interval = 500,
    retryIntervalMultiplier = 1,
    successCondition = () => true,
    success = ({ res }: any) => res,
    errorCondition = () => false,
    error = ({ res, err }: any) => {
      throw new Error(res || err);
    },
  } = {},
) => {
  try {
    const fnResult = await fn();
    // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
    if (successCondition(fnResult)) return success({ res: fnResult });
    // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
    if (retries === 0 || errorCondition(fnResult)) return error({ res: fnResult });
    await new Promise((resolve) => setTimeout(resolve, interval));
    return retry(fn, {
      retries: retries - 1,
      interval: interval * retryIntervalMultiplier,
      retryIntervalMultiplier,
      success,
      error,
      successCondition,
      errorCondition,
    });
  } catch (err) {
    return error({ err });
  }
};

// not a 200 and also not a retryable HTTP status code
const defaultFetchErrorConditionFn = (res: any) => !res || (!res.ok && !HTTPRetryStatuses.includes(res.status));

/**
 * Makes a fetch request.
 * @param relativeUrl: relative URL to the shard URL
 * @param method: HTTP method for the request
 * @param body: request body
 * @param headerOptions: additional headers for the request
 * @param options: additional fetch options for the request
 * @param timeoutMs: timeout for the request in milliseconds, defaults to browser timeout
 * @param retries: Number of times to retry the request on a retryable HTTP status code
 * @param initialDelay: Initial delay for the retry, will be doubled for each additional retry
 * @param success: callback on 200 responses
 * @param error: callback on non 200 responses
 * note that fetch won't reject HTTP error status
 * so this is the callback to handle non 200 responses.
 * See https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API#differences_from_jquery
 * @returns {Promise<T>}
 */
export const fetchEndpoint = ({
  relativeUrl,
  method = HTTPMethods.GET,
  body = undefined,
  headerOptions = {},
  options = {},
  timeoutMs = undefined,
  retries = 7,
  initialDelay = 1000,
  success = defaultResponseParser,
  error = defaultError,
  errorCondition = defaultFetchErrorConditionFn,
}: any) => {
  return new Promise((resolve, reject) =>
    retry(
      () =>
        fetchEndpointRaw({
          relativeUrl,
          method,
          body,
          headerOptions,
          options,
          timeoutMs,
        }),
      {
        retries,
        interval: initialDelay,
        retryIntervalMultiplier: 2,
        // 200s
        // @ts-expect-error TS(2322): Type '(res: any) => any' is not assignable to type... Remove this comment to see the full error message
        successCondition: (res: any) => res && res.ok,
        success: ({ res }) => success({ resolve, reject, response: res }),
        errorCondition,
        // @ts-expect-error TS(2322): Type '({ res, err }: any) => any' is not assignabl... Remove this comment to see the full error message
        error: ({ res, err }) => error({ resolve, reject, response: res, err: err }),
      },
    ),
  );
};

const filterUndefinedFields = (data: any) => {
  if (!Array.isArray(data)) {
    return pickBy(data, (v) => v !== undefined);
  } else {
    return data.filter((v) => v !== undefined);
  }
};

// Generate request body from js object or a string
const generateJsonBody = (data: any) => {
  if (typeof data === 'string') {
    // assuming the input is already a valid JSON string
    return data;
  } else if (typeof data === 'object') {
    return JSON.stringify(filterUndefinedFields(data));
  } else {
    throw new Error(
      // Reported during ESLint upgrade
      // eslint-disable-next-line max-len
      'Unexpected type of input. The REST api payload type must be either an object or a string, got ' + typeof data,
    );
  }
};

/* All functions below are essentially syntactic sugars for fetchEndpoint */

export const getJson = (props: any) => {
  const { relativeUrl, data } = props;
  const queryParams = new URLSearchParams(filterUndefinedFields(data)).toString();
  const combinedUrl = queryParams ? `${relativeUrl}?${queryParams}` : relativeUrl;
  return fetchEndpoint({
    ...props,
    relativeUrl: combinedUrl,
    method: HTTPMethods.GET,
    success: defaultResponseParser,
  });
};

export const postJson = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    method: HTTPMethods.POST,
    body: generateJsonBody(data),
    success: defaultResponseParser,
    ...props,
  });
};

export const putJson = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.PUT,
    body: generateJsonBody(data),
    success: defaultResponseParser,
  });
};

export const patchJson = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.PATCH,
    body: generateJsonBody(data),
    success: defaultResponseParser,
  });
};

export const deleteJson = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.DELETE,
    body: generateJsonBody(data),
    success: defaultResponseParser,
  });
};

export const getBigIntJson = (props: any) => {
  const { relativeUrl, data } = props;
  const queryParams = new URLSearchParams(filterUndefinedFields(data));
  return fetchEndpoint({
    ...props,
    ...(String(queryParams).length > 0 && {
      relativeUrl: `${relativeUrl}?${queryParams}`,
    }),
    method: HTTPMethods.GET,
    success: jsonBigIntResponseParser,
  });
};

export const postBigIntJson = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.POST,
    body: generateJsonBody(data),
    success: jsonBigIntResponseParser,
  });
};

export const putBigIntJson = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.PUT,
    body: generateJsonBody(data),
    success: jsonBigIntResponseParser,
  });
};

export const patchBigIntJson = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.PATCH,
    body: generateJsonBody(data),
    success: jsonBigIntResponseParser,
  });
};

export const deleteBigIntJson = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.DELETE,
    body: generateJsonBody(data),
    success: jsonBigIntResponseParser,
  });
};

export const getYaml = (props: any) => {
  const { relativeUrl, data } = props;
  const queryParams = new URLSearchParams(filterUndefinedFields(data));
  return fetchEndpoint({
    ...props,
    ...(queryParams && { relativeUrl: `${relativeUrl}?${queryParams}` }),
    method: HTTPMethods.GET,
    success: yamlResponseParser,
  });
};

export const postYaml = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.POST,
    body: generateJsonBody(data),
    success: yamlResponseParser,
  });
};

export const putYaml = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.PUT,
    body: generateJsonBody(data),
    success: yamlResponseParser,
  });
};

export const patchYaml = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.PATCH,
    body: generateJsonBody(data),
    success: yamlResponseParser,
  });
};

export const deleteYaml = (props: any) => {
  const { data } = props;
  return fetchEndpoint({
    ...props,
    method: HTTPMethods.DELETE,
    body: generateJsonBody(data),
    success: yamlResponseParser,
  });
};

function serializeRequestBody(payload: any | FormData | Blob) {
  if (payload === undefined) {
    return undefined;
  }
  return typeof payload === 'string' || payload instanceof FormData || payload instanceof Blob
    ? payload
    : JSON.stringify(payload);
}

export const fetchAPI = async (url: string, method: 'POST' | 'GET' | 'PATCH' | 'DELETE' = 'GET', body?: any) => {
  const response = await fetch(url, {
    method,
    body: serializeRequestBody(body),
    headers: body ? { 'Content-Type': 'application/json' } : {},
  });
  if (!response.ok) {
    const predefinedError = matchPredefinedError(response);
    if (predefinedError) {
      try {
        // Attempt to use message from the response
        const message = (await response.json()).message;
        predefinedError.message = message ?? predefinedError.message;
      } catch {
        // If the message can't be parsed, use default one
      }
      throw predefinedError;
    }
  }
  return response.json();
};

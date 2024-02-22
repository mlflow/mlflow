/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import {
  defaultResponseParser,
  getDefaultHeadersFromCookies,
  HTTPMethods,
  HTTPRetryStatuses,
  jsonBigIntResponseParser,
  parseResponse,
  yamlResponseParser,
  retry,
  fetchEndpointRaw,
  fetchEndpoint,
  getJson,
  getBigIntJson,
  putBigIntJson,
  patchBigIntJson,
  getYaml,
  putJson,
  putYaml,
  patchJson,
  patchYaml,
  postJson,
  postBigIntJson,
  postYaml,
  deleteJson,
  deleteBigIntJson,
  deleteYaml,
} from './FetchUtils';
import { ErrorWrapper } from './ErrorWrapper';

describe('FetchUtils', () => {
  describe('getDefaultHeadersFromCookies', () => {
    it('empty cookie should result in no headers', () => {
      expect(getDefaultHeadersFromCookies('')).toEqual({});
    });
    it('cookies from static service are parsed correctly', () => {
      expect(
        getDefaultHeadersFromCookies(`a=b; mlflow-request-header-My-CSRF=1; mlflow-request-header-Hello=World; c=d`),
      ).toEqual({ 'My-CSRF': '1', Hello: 'World' });
    });
  });
  describe('parseResponse', () => {
    let mockResolve: any;
    beforeEach(() => {
      mockResolve = jest.fn();
    });
    afterEach(() => {
      jest.clearAllMocks();
    });
    it('parseResponse parser failure resolves to text', async () => {
      const invalidParser = () => {
        throw new Error('failed to parse');
      };
      const mockResponse = {
        text: () => Promise.resolve('text'),
      };
      parseResponse({ resolve: mockResolve, response: mockResponse, parser: invalidParser });
      await new Promise(setImmediate);
      expect(mockResolve).toHaveBeenCalledWith('text');
    });
    it('defaultResponseParser', async () => {
      const mockResponse = {
        text: () => Promise.resolve('{"a": 123, "b": "flying monkey"}'),
      };
      defaultResponseParser({ resolve: mockResolve, response: mockResponse });
      await new Promise(setImmediate);
      expect(mockResolve).toHaveBeenCalledWith({ a: 123, b: 'flying monkey' });
    });
    it('jsonBigIntResponseParser', async () => {
      const mockResponse = {
        text: () => Promise.resolve('{"a": 11111111222222223333333344444445555555555}'),
      };
      jsonBigIntResponseParser({ resolve: mockResolve, response: mockResponse });
      await new Promise(setImmediate);
      expect(mockResolve).toHaveBeenCalledWith({ a: '11111111222222223333333344444445555555555' });
    });
    it('yamlResponseParser', async () => {
      const mockResponse = {
        text: () => Promise.resolve('artifact_path: model_signature\nflavors:\n  keras:\n    data: data\n'),
      };
      yamlResponseParser({ resolve: mockResolve, response: mockResponse });
      await new Promise(setImmediate);
      expect(mockResolve).toHaveBeenCalledWith({
        artifact_path: 'model_signature',
        flavors: { keras: { data: 'data' } },
      });
    });
  });
  describe('fetchEndpointRaw', () => {
    let mockResponse: any;
    let setTimeoutSpy: any;
    let mockFetch: any;
    let relativeUrl: any;
    let mockData: any;
    beforeEach(() => {
      mockResponse = {
        ok: true,
        status: 200,
        statusText: 'Ok',
        text: () => Promise.resolve('{"crazy": 8}'),
      };
      mockFetch = jest.fn(() => Promise.resolve(mockResponse));
      global.fetch = mockFetch;
      setTimeoutSpy = jest.spyOn(window, 'setTimeout');
      relativeUrl = '/ajax-api/2.0/service/endpoint';
      mockData = {
        group_id: 12345,
        user_id: 'qwerty',
        experimental_user: true,
        invalid_field: undefined,
      };
    });
    afterEach(() => {
      jest.clearAllMocks();
    });
    it('default headerOptions and options are expected', () => {
      Object.values(HTTPMethods).forEach(async (method) => {
        await fetchEndpointRaw({ relativeUrl, method, data: mockData });
        expect(mockFetch).toHaveBeenCalledWith(relativeUrl, {
          dataType: 'json',
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
          method,
        });
      });
    });
    it('overridden headerOptions and options are propagated correctly', () => {
      const customHeaders = { zzz_header: '123456', 'Content-Type': 'application/text' };
      const customOptions = { redirect: 'follow', dataType: 'text' };
      Object.values(HTTPMethods).forEach(async (method) => {
        await fetchEndpointRaw({
          relativeUrl,
          method,
          data: mockData,
          headerOptions: customHeaders,
          options: customOptions,
        });
        expect(mockFetch).toHaveBeenCalledWith(relativeUrl, {
          dataType: 'text',
          headers: { 'Content-Type': 'application/text', zzz_header: '123456' },
          method,
          redirect: 'follow',
        });
      });
    });
    it('setting timeout triggers setTimeout', async () => {
      await fetchEndpointRaw({ relativeUrl, timeoutMs: 1000 });
      // setTimeout should have been called
      expect(setTimeoutSpy).toHaveBeenCalled();
      // abort signal should've been set by abort controller
      expect(mockFetch.mock.calls[0][1].signal).not.toBeUndefined();
    });
    it('without setting timeout setTimeout is not triggered', async () => {
      await fetchEndpointRaw({ relativeUrl });
      // setTimeout should NOT have been called
      expect(setTimeoutSpy).not.toHaveBeenCalled();
      // signal should be undefined
      expect(mockFetch.mock.calls[0][1].signal).toBeUndefined();
    });
  });
  describe('retry', () => {
    let mockFn: any;
    let mockSuccess: any;
    let mockError: any;
    let mockSuccessCondition: any;
    let mockErrorCondition: any;
    let setTimeoutSpy: any;
    const retryWithMocks = ({ retries = 5, interval = 5, retryIntervalMultiplier = 1 }) =>
      retry(mockFn, {
        retries: retries,
        interval: interval,
        retryIntervalMultiplier: retryIntervalMultiplier,
        successCondition: mockSuccessCondition,
        success: mockSuccess,
        errorCondition: mockErrorCondition,
        error: mockError,
      });
    beforeEach(() => {
      mockFn = jest.fn(() => 'response string');
      mockSuccess = jest.fn();
      mockError = jest.fn();
      mockSuccessCondition = jest.fn(() => true);
      mockErrorCondition = jest.fn(() => false);
      setTimeoutSpy = jest.spyOn(window, 'setTimeout');
    });
    afterEach(() => {
      jest.clearAllMocks();
    });
    it('retry triggers success callback on successCondition=true', async () => {
      mockSuccessCondition = jest.fn(() => true);
      mockErrorCondition = jest.fn(() => false);
      await retryWithMocks({});
      // trigger fn and no timeout
      expect(mockFn).toHaveBeenCalledTimes(1);
      expect(setTimeoutSpy).not.toHaveBeenCalled();
      // success callback is triggered with response
      expect(mockSuccess).toHaveBeenCalledTimes(1);
      expect(mockSuccess).toHaveBeenCalledWith({ res: 'response string' });
      // error callback is not triggered
      expect(mockError).not.toHaveBeenCalled();
    });
    it('retry triggers error callback on errorCondition=true', async () => {
      mockSuccessCondition = jest.fn(() => false);
      mockErrorCondition = jest.fn(() => true);
      await retryWithMocks({});
      // trigger fn and no timeout
      expect(mockFn).toHaveBeenCalledTimes(1);
      expect(setTimeoutSpy).not.toHaveBeenCalled();
      // error callback is triggered with response
      expect(mockError).toHaveBeenCalledTimes(1);
      expect(mockError).toHaveBeenCalledWith({ res: 'response string' });
      // success callback is not triggered
      expect(mockSuccess).not.toHaveBeenCalled();
    });
    it('retry triggers success callback when both successCondition and errorCondition are true', async () => {
      mockSuccessCondition = jest.fn(() => true);
      mockErrorCondition = jest.fn(() => true);
      await retryWithMocks({});
      // trigger fn and no timeout
      expect(mockFn).toHaveBeenCalledTimes(1);
      expect(setTimeoutSpy).not.toHaveBeenCalled();
      // success callback is triggered with response
      expect(mockSuccess).toHaveBeenCalledTimes(1);
      expect(mockSuccess).toHaveBeenCalledWith({ res: 'response string' });
      // error callback is not triggered
      expect(mockError).not.toHaveBeenCalled();
    });
    it('retry triggers error callback on exception', async () => {
      mockFn = jest.fn(() => {
        throw new Error('oops');
      });
      await retryWithMocks({});
      expect(mockFn).toHaveBeenCalledTimes(1);
      expect(setTimeoutSpy).not.toHaveBeenCalled();
      // error callback is triggered with the err
      expect(mockError).toHaveBeenCalledTimes(1);
      expect(mockError).toHaveBeenCalledWith({ err: new Error('oops') });
      // success callback is not triggered
      expect(mockSuccess).not.toHaveBeenCalled();
    });
    it('retry triggers success callback after satisfying successCondition within retry limits', async () => {
      let n = 0;
      // make the function success after 3 retries
      mockSuccessCondition = jest.fn(() => {
        const res = n === 3;
        n += 1;
        return res;
      });
      mockErrorCondition = jest.fn(() => false);
      await retryWithMocks({});
      // function is triggered for 4 times and we've set timeout for 3 times
      expect(mockFn).toHaveBeenCalledTimes(4);
      expect(setTimeoutSpy).toHaveBeenCalledTimes(3);
      // success callback is triggered after 3 retries with the response
      expect(mockSuccess).toHaveBeenCalledTimes(1);
      expect(mockSuccess).toHaveBeenCalledWith({ res: 'response string' });
      // error callback is not triggered
      expect(mockError).not.toHaveBeenCalled();
    });
    it('retry triggers error callback after reaching retry limits', async () => {
      mockSuccessCondition = jest.fn(() => false);
      mockErrorCondition = jest.fn(() => false);
      await retryWithMocks({});
      // function is triggered for 6 times and we've set timeout for 5 times
      expect(mockFn).toHaveBeenCalledTimes(6);
      expect(setTimeoutSpy).toHaveBeenCalledTimes(5);
      // error callback is triggered after 5 retries with the response
      expect(mockError).toHaveBeenCalledTimes(1);
      expect(mockError).toHaveBeenCalledWith({ res: 'response string' });
      // success callback is not triggered
      expect(mockSuccess).not.toHaveBeenCalled();
    });
    it('retryIntervalMultiplier changes retry interval', async () => {
      mockSuccessCondition = jest.fn(() => false);
      mockErrorCondition = jest.fn(() => false);
      await retryWithMocks({ retries: 5, interval: 100, retryIntervalMultiplier: 2 });
      expect(mockFn).toHaveBeenCalledTimes(6);
      let i = 0;
      [100, 200, 400, 800, 1600].forEach((interval) => {
        expect(setTimeoutSpy.mock.calls[i][1]).toEqual(interval);
        i += 1;
      });
    });
  });
  describe('fetchEndpoint', () => {
    it('fetchEndpoint resolves on ok response', async () => {
      const okResponse = { ok: true, status: 200, text: () => Promise.resolve('{"dope": "ape"}') };
      // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: boolean; status: number; ... Remove this comment to see the full error message
      global.fetch = jest.fn(() => Promise.resolve(okResponse));
      await expect(fetchEndpoint({ relativeUrl: 'http://localhost:3000' })).resolves.toEqual({
        dope: 'ape',
      });
    });
    it.each(HTTPRetryStatuses)(
      'fetchEndpoint resolves on consecutive retry status and a final valid response',
      async (retryStatus) => {
        const tooManyRequestsResponse = {
          ok: false,
          status: retryStatus,
          statusText: 'TooManyRequests',
        };
        const okResponse = {
          ok: true,
          status: 200,
          text: () => Promise.resolve('{"sorry": "I am late"}'),
        };
        const responses = [...Array(2).fill(tooManyRequestsResponse), okResponse];
        // pop the head of the array on each call
        global.fetch = jest.fn(() => Promise.resolve(responses.shift()));
        await expect(
          fetchEndpoint({
            relativeUrl: 'http://localhost:3000',
            initialDelay: 5,
            retries: 4,
          }),
        ).resolves.toEqual({ sorry: 'I am late' });
      },
    );
    it.each(HTTPRetryStatuses)(
      'fetchEndpoint rejects on consecutive retry status and no valid response',
      async (retryStatus) => {
        const tooManyRequestsResponse = {
          ok: false,
          status: retryStatus,
          text: () => Promise.resolve('{error_code: "TooManyRequests", message: "TooManyRequests"}'),
        };
        const responses = Array(3).fill(tooManyRequestsResponse);
        global.fetch = jest.fn(() => Promise.resolve(responses.shift()));
        await expect(
          fetchEndpoint({
            relativeUrl: 'http://localhost:3000',
            initialDelay: 5,
            retries: 2,
          }),
        ).rejects.toEqual(new ErrorWrapper('{error_code: "TooManyRequests", message: "TooManyRequests"}', retryStatus));
      },
    );
    it('fetchEndpoint rejects on non retry status failures', async () => {
      const permissionDeniedResponse = {
        ok: false,
        status: 403,
        text: () => Promise.resolve('{error_code: "PermissionDenied", message: "PermissionDenied"}'),
      };
      // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: boolean; status: number; ... Remove this comment to see the full error message
      global.fetch = jest.fn(() => Promise.resolve(permissionDeniedResponse));
      await expect(
        fetchEndpoint({
          relativeUrl: 'http://localhost:3000',
          initialDelay: 5,
          retries: 2,
        }),
      ).rejects.toEqual(new ErrorWrapper('{error_code: "PermissionDenied", message: "PermissionDenied"}', 403));
    });
    it('fetchEndpoint rejects on random exceptions', async () => {
      const randomError = new Error('something went wrong...');
      global.fetch = jest.fn(() => Promise.reject(randomError));
      await expect(
        fetchEndpoint({
          relativeUrl: 'http://localhost:3000',
          initialDelay: 5,
          retries: 2,
        }),
      ).rejects.toEqual(new ErrorWrapper(new Error('something went wrong...'), 500));
    });
  });
  describe('fetchEndpoint syntactic sugars', () => {
    let mockResponse: any;
    let mockFetch: any;
    let relativeUrl: any;
    let mockData: any;
    beforeEach(() => {
      mockResponse = {
        ok: true,
        status: 200,
        statusText: 'Ok',
        text: () => Promise.resolve('{"crazy": 8}'),
      };
      mockFetch = jest.fn(() => Promise.resolve(mockResponse));
      global.fetch = mockFetch;
      relativeUrl = '/ajax-api/2.0/service/endpoint';
      mockData = {
        group_id: 12345,
        user_id: 'qwerty',
        experimental_user: false,
        null_field: null,
        undefined_field: undefined,
      };
    });
    afterEach(() => {
      jest.clearAllMocks();
    });
    it('GET requests bake data in query params', () => {
      [getJson, getBigIntJson, getYaml].forEach(async (getCall) => {
        await getCall({ relativeUrl, data: mockData });
        expect(mockFetch).toHaveBeenCalledWith(
          `${relativeUrl}?group_id=12345&user_id=qwerty&experimental_user=false&null_field=null`,
          {
            dataType: 'json',
            headers: { 'Content-Type': 'application/json; charset=utf-8' },
            method: 'GET',
          },
        );
      });
    });
    it('other requests pass data to request body', () => {
      [
        { fetchCall: putJson, method: HTTPMethods.PUT },
        { fetchCall: putBigIntJson, method: HTTPMethods.PUT },
        { fetchCall: putYaml, method: HTTPMethods.PUT },
        { fetchCall: patchJson, method: HTTPMethods.PATCH },
        { fetchCall: patchBigIntJson, method: HTTPMethods.PATCH },
        { fetchCall: patchYaml, method: HTTPMethods.PATCH },
        { fetchCall: postJson, method: HTTPMethods.POST },
        { fetchCall: postBigIntJson, method: HTTPMethods.POST },
        { fetchCall: postYaml, method: HTTPMethods.POST },
        { fetchCall: deleteJson, method: HTTPMethods.DELETE },
        { fetchCall: deleteBigIntJson, method: HTTPMethods.DELETE },
        { fetchCall: deleteYaml, method: HTTPMethods.DELETE },
      ].forEach((args) => {
        const { fetchCall, method } = args;
        const mockArrayData = [1, undefined, null, 2];
        const mockStringData = '[1, undefined, null, 2]';
        fetchCall({ relativeUrl, data: mockData });
        expect(mockFetch).toHaveBeenLastCalledWith(relativeUrl, {
          dataType: 'json',
          body: JSON.stringify({
            group_id: 12345,
            user_id: 'qwerty',
            experimental_user: false,
            null_field: null,
          }),
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
          method,
        });
        fetchCall({ relativeUrl, data: mockArrayData });
        expect(mockFetch).toHaveBeenLastCalledWith(relativeUrl, {
          dataType: 'json',
          body: JSON.stringify([1, null, 2]),
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
          method,
        });
        fetchCall({ relativeUrl, data: mockStringData });
        expect(mockFetch).toHaveBeenCalledWith(relativeUrl, {
          dataType: 'json',
          body: mockStringData,
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
          method,
        });
      });
    });
  });
});

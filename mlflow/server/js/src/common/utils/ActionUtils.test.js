import { ErrorWrapper, wrapDeferred } from './ActionUtils';

/**
 * Returns mock Ajax function that sequentially responds with status codes in `responseStatusCodes`,
 * looping back to the start of the array of status codes if necessary.
 */
const getMockAjax = (responseStatusCodes) => {
  let i = 0;
  return ({ success, error }) => {
    const statusCode = responseStatusCodes[i % responseStatusCodes.length];
    if (statusCode === 200) {
      success({ requestIndex: i });
    } else {
      error({ status: statusCode });
    }
    i += 1;
  };
};

describe('ErrorWrapper', () => {
  test('renderHttpError works on DatabricksServiceExceptions', () => {
    const xhr = {
      responseText:
        '{ "error_code": "INVALID_REQUEST", "message": "Foo!", "stack_trace": "Boop!" }',
    };
    const error = new ErrorWrapper(xhr).renderHttpError();
    expect(error).toEqual('INVALID_REQUEST: Foo!');
  });

  test('renderHttpError works on HTML', () => {
    const xhr = {
      responseText: '<div>This\n\n\n</div>Is an error!<br/>',
    };
    const error = new ErrorWrapper(xhr).renderHttpError();
    expect(error).toEqual('This\nIs an error!');
  });

  test('renderHttpError works weird stuff', () => {
    const xhr = {};
    const error = new ErrorWrapper(xhr).renderHttpError();
    expect(error).toEqual('Request Failed');
  });

  test('ErrorWrapper.getErrorCode does not fail on JSON decoding problems', () => {
    new ErrorWrapper({ responseText: 'a{waefaw' });
  });
});

describe('wrapDeferred', () => {
  test('wrapDeferred retries on 429s', (done) => {
    wrapDeferred(getMockAjax([429, 429, 200]), {}, 1000, 10)
      .then((result) => {
        expect(result).toEqual({ requestIndex: 2 });
      })
      .then(() => done());
  });

  test('wrapDeferred responds with 429 after timeout period', (done) => {
    // First request responds with 429, we then retry and receive another 429, but don't have
    // time to retry again, so we propagate the 429
    let caughtError = false;
    wrapDeferred(getMockAjax([429, 429, 200]), {}, 10, 10)
      .catch((result) => {
        expect(result.xhr.status).toEqual(429);
        caughtError = true;
      })
      .then(() => {
        if (caughtError) {
          done();
        }
      });
  });

  test('wrapDeferred does not retry on 200s', (done) => {
    wrapDeferred(getMockAjax([200]), {}, 1000, 10)
      .then((result) => {
        expect(result).toEqual({ requestIndex: 0 });
      })
      .then(() => done());
  });

  test('wrapDeferred does not retry on non-429 error codes', (done) => {
    wrapDeferred(getMockAjax([503]), {}, 10, 10)
      .catch((result) => {
        expect(result.xhr.status).toEqual(503);
      })
      .then(() => done());
  });
});

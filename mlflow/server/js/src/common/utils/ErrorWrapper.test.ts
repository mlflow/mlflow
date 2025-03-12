import { ErrorWrapper } from './ErrorWrapper';

describe('ErrorWrapper', () => {
  it('renderHttpError works on DatabricksServiceExceptions', () => {
    const error = new ErrorWrapper('{ "error_code": "INVALID_REQUEST", "message": "Foo!" }', 400).renderHttpError();
    expect(error).toEqual('INVALID_REQUEST: Foo!');
  });

  it('renderHttpError works on DatabricksServiceExceptions with stack traces', () => {
    const error = new ErrorWrapper(
      '{ "error_code": "INVALID_REQUEST", "message": "Foo!", "stack_trace": "Boop!" }',
      400,
    ).renderHttpError();
    expect(error).toEqual('INVALID_REQUEST: Foo!\n\nBoop!');
  });

  it('renderHttpError works on HTML', () => {
    const error = new ErrorWrapper('<div>This\n\n\n</div>Is an error!<br/>', 400).renderHttpError();
    expect(error).toEqual('This\nIs an error!');
  });

  it('renderHttpError works weird stuff', () => {
    const error = new ErrorWrapper('{}', 500).renderHttpError();
    expect(error).toEqual('Request Failed');
  });

  it('ErrorWrapper.getErrorCode does not fail on JSON decoding problems', () => {
    const error = new ErrorWrapper('a{waefaw', 400).getErrorCode();
    expect(error).toEqual('INTERNAL_ERROR');
  });

  it('ErrorWrapper.is4xxError correctly detects 4XX error', () => {
    const error401 = new ErrorWrapper('{}', 401);
    const error404 = new ErrorWrapper('{}', 404);
    const error503 = new ErrorWrapper('{}', 503);
    const errorUncategorized = new ErrorWrapper('some textual error');

    expect(error401.is4xxError()).toEqual(true);
    expect(error404.is4xxError()).toEqual(true);
    expect(error503.is4xxError()).toEqual(false);
    expect(errorUncategorized.is4xxError()).toEqual(false);
  });
});

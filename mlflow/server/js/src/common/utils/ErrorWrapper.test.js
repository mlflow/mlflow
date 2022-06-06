import { ErrorWrapper } from './ErrorWrapper';

describe('ErrorWrapper', () => {
  it('renderHttpError works on DatabricksServiceExceptions', () => {
    const error = new ErrorWrapper(
      '{ "error_code": "INVALID_REQUEST", "message": "Foo!" }',
      400,
    ).renderHttpError();
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
    const error = new ErrorWrapper('{}', '500').renderHttpError();
    expect(error).toEqual('Request Failed');
  });

  it('ErrorWrapper.getErrorCode does not fail on JSON decoding problems', () => {
    const error = new ErrorWrapper('a{waefaw', 400).getErrorCode();
    expect(error).toEqual('INTERNAL_ERROR');
  });
});

/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

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
    // @ts-expect-error TS(2345): Argument of type 'string' is not assignable to par... Remove this comment to see the full error message
    const error = new ErrorWrapper('{}', '500').renderHttpError();
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

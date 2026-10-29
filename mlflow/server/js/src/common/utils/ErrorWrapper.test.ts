import { describe, it, expect } from '@jest/globals';
import { ErrorWrapper } from './ErrorWrapper';

describe('ErrorWrapper', () => {
  it('prompts a stale tab to reload only for a marked artifact-mode rejection', () => {
    const error = new ErrorWrapper(
      {
        error_code: 'PERMISSION_DENIED',
        message: 'This server requires artifacts to be downloaded directly from storage.',
        artifacts_presigned_only: true,
      },
      403,
    );
    expect(error.renderHttpError()).toBe('The server artifact settings have changed. Reload this page to continue.');
    expect(error.getMessageField()).toBe(error.renderHttpError());
    expect(error.translateToErrorInstance().message).toBe(error.renderHttpError());
  });

  it('preserves ordinary permission errors instead of prompting a reload', () => {
    const error = new ErrorWrapper({ error_code: 'PERMISSION_DENIED', message: 'Access denied' }, 403);
    expect(error.renderHttpError()).toBe('PERMISSION_DENIED: Access denied');
    expect(error.getMessageField()).toBe('Access denied');
  });

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

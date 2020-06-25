import { ErrorWrapper } from './ActionUtils';

test('renderHttpError works on DatabricksServiceExceptions', () => {
  const xhr = {
    responseText: '{ "error_code": "INVALID_REQUEST", "message": "Foo!", "stack_trace": "Boop!" }',
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

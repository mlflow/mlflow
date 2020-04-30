import { getRequestHeaders } from './setupAjaxHeaders';

test('empty cookie should result in no headers', () => {
  const headers = getRequestHeaders('');
  expect(headers).toEqual({});
});

test('cookies prefixed with mlflow-request-header- should be returned', () => {
  const headers = getRequestHeaders(
    'a=b; mlflow-request-header-My-CSRF=1; mlflow-request-header-Hello=World; c=d',
  );
  expect(headers).toEqual({ 'My-CSRF': '1', Hello: 'World' });
});

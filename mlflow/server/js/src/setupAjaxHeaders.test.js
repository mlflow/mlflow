import getRequestHeaders from './setupAjaxHeaders';

test('If activeExperimentId is defined then choose that one', () => {
  const headers = getRequestHeaders("");
  expect(headers).toEqual({});
});

test('If activeExperimentId is defined then choose that one', () => {
  const headers = getRequestHeaders(
    "a=b; mlflow-request-header-My-CSRF=1; mlflow-request-header-Hello=World; c=d");
  expect(headers).toEqual({"My-CSRF": "1", "Hello": "World"});
});

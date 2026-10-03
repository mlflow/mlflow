import { afterAll, afterEach, beforeAll, describe, expect, test } from '@jest/globals';
import { rest } from 'msw';
import { setupServer } from '../common/utils/setup-msw';
import { getModelVersionArtifactApi, resolveFilterValue } from './actions';

describe('action tests', () => {
  test('simple string', () => {
    expect(resolveFilterValue('hello')).toEqual("'hello'");
  });

  test('simple string with wildcard', () => {
    expect(resolveFilterValue('hello', true)).toEqual("'%hello%'");
  });

  test('simple string spaces', () => {
    expect(resolveFilterValue(' he llo  ')).toEqual("' he llo  '");
  });

  test('simple string spaces with wildcard', () => {
    expect(resolveFilterValue(' he llo  ', true)).toEqual("'% he llo  %'");
  });

  test('single quotes', () => {
    expect(resolveFilterValue("A's model")).toEqual('"A\'s model"');
  });

  test('single quotes with wildcard', () => {
    expect(resolveFilterValue("A's model", true)).toEqual('"%A\'s model%"');
  });

  test('double quotes', () => {
    expect(resolveFilterValue('the "best" model')).toEqual('\'the "best" model\'');
  });

  test('double quotes with wildcard', () => {
    expect(resolveFilterValue('the "best" model', true)).toEqual('\'%the "best" model%\'');
  });

  test('percent character', () => {
    expect(resolveFilterValue('%')).toEqual("'%'");
  });

  test('percent character with wildcard', () => {
    expect(resolveFilterValue('%', true)).toEqual("'%%%'");
  });
});

describe('getModelVersionArtifactApi', () => {
  const proxyAnchor = 'http://localhost/api/2.0/mlflow-artifacts/artifacts';

  let capturedUrls: string[] = [];
  const server = setupServer(
    rest.get(/get-artifact|mlflow-artifacts/, (req, res, ctx) => {
      capturedUrls.push(req.url.toString());
      return res(ctx.body('mlmodel-content'));
    }),
  );

  beforeAll(() => server.listen());

  afterEach(() => {
    capturedUrls = [];
    server.resetHandlers();
  });

  afterAll(() => server.close());

  test('reads the MLmodel file from an eligible artifact proxy source', async () => {
    await getModelVersionArtifactApi('my-model', '1', undefined, `${proxyAnchor}/model-root`).payload;

    expect(capturedUrls).toEqual([`${proxyAnchor}/model-root/MLmodel`]);
  });

  test('reads the MLmodel file through the tracking server when the source is not eligible', async () => {
    await getModelVersionArtifactApi('my-model', '1', undefined, 's3://bucket/model-root').payload;

    expect(capturedUrls).toHaveLength(1);
    expect(capturedUrls[0]).toContain('model-versions/get-artifact');
  });

  test('reads the MLmodel file through the tracking server when no source is known', async () => {
    await getModelVersionArtifactApi('my-model', '1').payload;

    expect(capturedUrls[0]).toContain('model-versions/get-artifact');
  });
});

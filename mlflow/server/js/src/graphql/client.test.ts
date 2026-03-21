import 'whatwg-fetch';
import { afterAll, beforeEach, describe, expect, it, jest } from '@jest/globals';

import { graphqlFetch } from './client';

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  getAjaxUrl: jest.fn(),
}));

const fetchUtils = jest.requireMock<typeof import('@mlflow/mlflow/src/common/utils/FetchUtils')>(
  '@mlflow/mlflow/src/common/utils/FetchUtils',
);
const getAjaxUrl = jest.mocked(fetchUtils.getAjaxUrl);

describe('graphqlFetch', () => {
  const originalFetch = global.fetch;
  const fetchMock = jest.fn<typeof global.fetch>();

  beforeEach(() => {
    fetchMock.mockResolvedValue({ ok: true } as Response);
    global.fetch = fetchMock;
    getAjaxUrl.mockReset();
  });

  afterAll(() => {
    global.fetch = originalFetch;
  });

  it('resolves graphql requests via the ajax url helper', async () => {
    const resolvedUrl = '/graphql';
    getAjaxUrl.mockImplementation(() => resolvedUrl);

    await graphqlFetch('graphql', { headers: { 'X-Test': '1' } });

    expect(getAjaxUrl).toHaveBeenCalledWith('graphql');
    expect(fetchMock).toHaveBeenCalledWith(
      resolvedUrl,
      expect.objectContaining({
        headers: expect.any(Headers),
      }),
    );
  });
});

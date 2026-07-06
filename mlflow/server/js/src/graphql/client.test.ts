import { afterAll, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { getAjaxUrl } from '../common/utils/FetchUtils';

import { graphqlFetch } from './client';

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FetchUtils')>(
    '@mlflow/mlflow/src/common/utils/FetchUtils',
  ),
  getAjaxUrl: jest.fn(),
}));

jest.mock('../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../common/utils/FeatureUtils')>('../common/utils/FeatureUtils'),
  shouldEnableSpogFetchPipeline: jest.fn().mockReturnValue(false),
}));
const mockedGetAjaxUrl = jest.mocked(getAjaxUrl);

describe('graphqlFetch', () => {
  const originalFetch = global.fetch;
  const fetchMock = jest.fn<typeof global.fetch>();

  beforeEach(() => {
    fetchMock.mockResolvedValue({ ok: true } as Response);
    global.fetch = fetchMock;
    mockedGetAjaxUrl.mockReset();
  });

  afterAll(() => {
    global.fetch = originalFetch;
  });

  it('resolves graphql requests via window.fetch when SPOG flag is off', async () => {
    const resolvedUrl = '/graphql';
    mockedGetAjaxUrl.mockImplementation(() => resolvedUrl);

    await graphqlFetch('graphql', { headers: { 'X-Test': '1' } });

    expect(mockedGetAjaxUrl).toHaveBeenCalledWith('graphql');
    expect(fetchMock).toHaveBeenCalledWith(
      resolvedUrl,
      expect.objectContaining({
        headers: expect.any(Headers),
      }),
    );
  });
});

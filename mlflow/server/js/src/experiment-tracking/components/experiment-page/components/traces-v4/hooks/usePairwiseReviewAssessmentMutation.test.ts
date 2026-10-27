import { describe, expect, jest, test } from '@jest/globals';

import { GenericNetworkRequestError } from '@databricks/web-shared/errors';

import { retryPairwiseAssessmentWrite } from './usePairwiseReviewAssessmentMutation';

describe('retryPairwiseAssessmentWrite', () => {
  test('retries only the failed assessment write', async () => {
    const baselineWrite = jest.fn<() => Promise<string>>().mockResolvedValue('baseline');
    const candidateWrite = jest
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new GenericNetworkRequestError({ status: 503 }))
      .mockResolvedValue('candidate');

    // The first retry waits 200 ms, so real timers keep this test short.
    await expect(
      Promise.all([retryPairwiseAssessmentWrite(baselineWrite), retryPairwiseAssessmentWrite(candidateWrite)]),
    ).resolves.toEqual(['baseline', 'candidate']);
    expect(baselineWrite).toHaveBeenCalledTimes(1);
    expect(candidateWrite).toHaveBeenCalledTimes(2);
  });

  test('does not retry a write that fails with a client error', async () => {
    const write = jest.fn<() => Promise<string>>().mockRejectedValue(new GenericNetworkRequestError({ status: 400 }));

    await expect(retryPairwiseAssessmentWrite(write)).rejects.toBeInstanceOf(GenericNetworkRequestError);
    expect(write).toHaveBeenCalledTimes(1);
  });
});

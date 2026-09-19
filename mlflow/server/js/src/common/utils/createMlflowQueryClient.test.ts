import { describe, test, expect } from '@jest/globals';
import { createMlflowQueryClient } from './createMlflowQueryClient';
import { onlineManager } from './reactQueryHooks';

describe('createMlflowQueryClient', () => {
  test('defaults queries to networkMode "always"', () => {
    expect(createMlflowQueryClient().getDefaultOptions().queries?.networkMode).toBe('always');
  });

  test('runs a query while the browser reports being offline', async () => {
    const queryClient = createMlflowQueryClient();
    onlineManager.setOnline(false);

    try {
      // With React Query's default networkMode this never resolves: the query is paused before
      // queryFn runs, which is what left the UI stuck on a loading skeleton.
      await expect(
        queryClient.fetchQuery({ queryKey: ['offline-probe'], queryFn: async () => 'resolved' }),
      ).resolves.toBe('resolved');
    } finally {
      onlineManager.setOnline(true);
      onlineManager.setOnline(undefined);
      queryClient.clear();
    }
  });
});

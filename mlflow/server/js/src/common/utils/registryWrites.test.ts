import { describe, expect, it } from '@jest/globals';
import { NotFoundError } from '@databricks/web-shared/errors';

import { ignoreNotFound, settleAll } from './registryWrites';

describe('registryWrites', () => {
  it('waits for every write before reporting the first failure', async () => {
    let lateWriteDone = false;
    const late = new Promise<void>((resolve) =>
      setTimeout(() => {
        lateWriteDone = true;
        resolve();
      }, 20),
    );

    await expect(settleAll([Promise.reject(new Error('tag rejected')), late])).rejects.toThrow('tag rejected');
    expect(lateWriteDone).toBe(true);
    await expect(settleAll([Promise.resolve(), Promise.resolve()])).resolves.toBeUndefined();
  });

  it('treats deleting something already gone as done, and nothing else', async () => {
    await expect(ignoreNotFound(Promise.reject(new NotFoundError({})))).resolves.toBeUndefined();
    await expect(
      ignoreNotFound(Promise.reject(Object.assign(new Error('gone'), { name: 'NotFoundError' }))),
    ).resolves.toBeUndefined();
    await expect(ignoreNotFound(Promise.reject(new Error('Permission denied')))).rejects.toThrow('Permission denied');
  });
});

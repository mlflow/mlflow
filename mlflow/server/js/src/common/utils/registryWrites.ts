import { NotFoundError } from '@databricks/web-shared/errors';

/**
 * Waits for every write before failing, so a partial failure is reported only once the server has settled and a
 * refresh shows everything that did change.
 */
export const settleAll = async (writes: Promise<unknown>[]) => {
  const failure = (await Promise.allSettled(writes)).find(
    (result): result is PromiseRejectedResult => result.status === 'rejected',
  );
  if (failure) throw failure.reason;
};

/**
 * Treats deleting something that is already gone as done. Editors keep the change list they computed when they
 * opened, so retrying a partly failed save re-sends deletes that went through the first time.
 */
export const ignoreNotFound = (deletion: Promise<unknown>) =>
  deletion.catch((error: unknown) => {
    if (!(error instanceof NotFoundError) && (error as Error | undefined)?.name !== 'NotFoundError') throw error;
  });

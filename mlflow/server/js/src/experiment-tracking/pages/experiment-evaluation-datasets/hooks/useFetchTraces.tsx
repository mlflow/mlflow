import { useQuery } from '@tanstack/react-query';
import { getTrace } from '../../../utils/TraceUtils';
import { FETCH_TRACES_QUERY_KEY } from '../constants';

const MAX_PARALLEL_REQUESTS = 10;

async function fetchWithConcurrency<T>(
  ids: string[],
  fetchFn: (id: string) => Promise<T>,
  concurrency: number = MAX_PARALLEL_REQUESTS,
): Promise<T[]> {
  const results: T[] = [];
  const executing = new Set<Promise<void>>();

  for (const id of ids) {
    const promise = fetchFn(id)
      .then((result) => {
        results.push(result);
      })
      .finally(() => {
        executing.delete(promise);
      });

    executing.add(promise);

    if (executing.size >= concurrency) {
      await Promise.race(executing);
    }
  }

  await Promise.all(executing);
  return results;
}

// TODO: migrate this to the batch get traces API in a shared location when it is available
export const useFetchTraces = ({ traceIds }: { traceIds: string[] }) => {
  return useQuery({
    queryKey: [FETCH_TRACES_QUERY_KEY, traceIds],
    queryFn: async ({ queryKey: [, traceIds] }) => {
      const traces = await fetchWithConcurrency(traceIds as string[], getTrace);
      return traces;
    },
  });
};

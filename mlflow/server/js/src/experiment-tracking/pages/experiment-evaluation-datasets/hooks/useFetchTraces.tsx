import { useQuery } from '@tanstack/react-query';
import { getTrace } from '../../../utils/TraceUtils';
import { FETCH_TRACES_QUERY_KEY } from '../constants';
import { chunk } from 'lodash';

const MAX_PARALLEL_REQUESTS = 20;

// TODO: migrate this to the batch get traces API in a shared location when it is available
export const useFetchTraces = ({ traceIds }: { traceIds: string[] }) => {
  return useQuery({
    queryKey: [FETCH_TRACES_QUERY_KEY, traceIds],
    queryFn: async ({ queryKey: [, traceIds] }) => {
      const chunks = chunk(traceIds, MAX_PARALLEL_REQUESTS);

      const results = [];
      for (const chunk of chunks) {
        const traces = await Promise.all(chunk.map(getTrace));
        results.push(...traces);
      }

      return results;
    },
  });
};

import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { MlflowService } from '../../../sdk/MlflowService';
import { EXPERIMENT_TRACES_SORTABLE_COLUMNS, getTraceInfoRunId } from '../TracesView.utils';
import { ViewType } from '../../../sdk/MlflowEnums';
import { first, uniq, values } from 'lodash';
import type { RunEntity } from '../../../types';

// A filter expression used to filter traces by run ID
const RUN_ID_FILTER_EXPRESSION = 'request_metadata.`mlflow.sourceRun`';
const LOGGED_MODEL_ID_FILTER_EXPRESSION = 'request_metadata.`mlflow.modelId`';

const createRunIdsFilterExpression = (runUuids: string[]) => {
  const runIdsInQuotes = runUuids.map((runId: any) => `'${runId}'`);
  return `run_id IN (${runIdsInQuotes.join(',')})`;
};

/**
 * Utility function that fetches run names for traces.
 */
const fetchRunNamesForTraces = async (experimentIds: string[], traces: ModelTraceInfo[]) => {
  const traceIdToRunIdMap = traces.reduce<Record<string, string>>((acc, trace) => {
    const traceId = trace.request_id;
    const runId = getTraceInfoRunId(trace);
    if (!traceId || !runId) {
      return acc;
    }
    return { ...acc, [traceId]: runId };
  }, {});

  const runUuids = uniq(values(traceIdToRunIdMap));
  if (runUuids.length < 1) {
    return {};
  }
  const runResponse = (await MlflowService.searchRuns({
    experiment_ids: experimentIds,
    filter: createRunIdsFilterExpression(runUuids),
    run_view_type: ViewType.ALL,
  })) as { runs?: RunEntity[] };

  const runs = runResponse.runs;

  const runIdsToRunNames = (runs || []).reduce<Record<string, string>>((acc, run) => {
    return { ...acc, [run.info.runUuid]: run.info.runName };
  }, {});

  const traceIdsToRunNames = traces.reduce<Record<string, string>>((acc, trace) => {
    const traceId = trace.request_id;
    if (!traceId) {
      return acc;
    }
    const runId = traceIdToRunIdMap[traceId];

    return { ...acc, [traceId]: runIdsToRunNames[runId] || runId };
  }, {});

  return traceIdsToRunNames;
};

export interface ModelTraceInfoWithRunName extends ModelTraceInfo {
  runName?: string;
}

export const useExperimentTraces = ({
  experimentIds,
  sorting,
  filter = '',
  runUuid,
  loggedModelId,
}: {
  experimentIds: string[];
  sorting: {
    id: string;
    desc: boolean;
  }[];
  filter?: string;
  runUuid?: string;
  loggedModelId?: string;
}) => {
  const [traces, setTraces] = useState<ModelTraceInfoWithRunName[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | undefined>(undefined);

  // Backend currently only supports ordering by timestamp
  const orderByString = useMemo(() => {
    const firstOrderByColumn = first(sorting);
    if (firstOrderByColumn && EXPERIMENT_TRACES_SORTABLE_COLUMNS.includes(firstOrderByColumn.id)) {
      return `${firstOrderByColumn.id} ${firstOrderByColumn.desc ? 'DESC' : 'ASC'}`;
    }
    return 'timestamp_ms DESC';
  }, [sorting]);

  const filterString = useMemo(() => {
    if (!runUuid && !loggedModelId) {
      return filter;
    }

    if (loggedModelId) {
      if (filter) {
        return `${filter} AND ${LOGGED_MODEL_ID_FILTER_EXPRESSION}='${loggedModelId}'`;
      }
      return `${LOGGED_MODEL_ID_FILTER_EXPRESSION}='${loggedModelId}'`;
    }

    if (filter) {
      return `${filter} AND ${RUN_ID_FILTER_EXPRESSION}='${runUuid}'`;
    }

    return `${RUN_ID_FILTER_EXPRESSION}='${runUuid}'`;
  }, [filter, runUuid, loggedModelId]);

  const [pageTokens, setPageTokens] = useState<Record<string, string | undefined>>({ 0: undefined });
  const [currentPage, setCurrentPage] = useState(0);
  const currentPageToken = pageTokens[currentPage];

  const fetchTraces = useCallback(
    async ({
      experimentIds,
      currentPage = 0,
      pageToken,
      silent,
      orderByString = '',
      filterString = '',
    }: {
      experimentIds: string[];
      currentPage?: number;
      pageToken?: string;
      filterString?: string;
      orderByString?: string;
      silent?: boolean;
    }) => {
      if (!silent) {
        setLoading(true);
      }
      setError(undefined);

      try {
        const response = await MlflowService.getExperimentTraces(experimentIds, orderByString, pageToken, filterString);

        if (!response.traces) {
          setTraces([]);
          return;
        }

        const runNamesForTraces = await fetchRunNamesForTraces(experimentIds, response.traces);
        const tracesWithRunNames = response.traces.map((trace) => {
          const traceId = trace.request_id;
          if (!traceId) {
            return { ...trace };
          }
          const runName = runNamesForTraces[traceId];
          return { ...trace, runName };
        });

        setTraces(tracesWithRunNames);
        setPageTokens((prevPages) => {
          return { ...prevPages, [currentPage + 1]: response.next_page_token };
        });
      } catch (e: any) {
        setError(e);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const hasNextPage = !loading && pageTokens[currentPage + 1] !== undefined;
  const hasPreviousPage = !loading && (currentPage === 1 || pageTokens[currentPage - 1] !== undefined);

  useEffect(() => {
    fetchTraces({ experimentIds, filterString, orderByString });
  }, [fetchTraces, filterString, experimentIds, orderByString]);

  const reset = useCallback(() => {
    setTraces([]);
    setPageTokens({ 0: undefined });
    setCurrentPage(0);
    fetchTraces({ experimentIds });
  }, [fetchTraces, experimentIds]);

  const fetchNextPage = useCallback(() => {
    setCurrentPage((prevPage) => prevPage + 1);
    fetchTraces({
      experimentIds,
      currentPage: currentPage + 1,
      pageToken: pageTokens[currentPage + 1],
      filterString,
      orderByString,
    });
  }, [experimentIds, currentPage, fetchTraces, pageTokens, filterString, orderByString]);

  const fetchPrevPage = useCallback(() => {
    setCurrentPage((prevPage) => prevPage - 1);
    fetchTraces({
      experimentIds,
      currentPage: currentPage - 1,
      pageToken: pageTokens[currentPage - 1],
      filterString,
      orderByString,
    });
  }, [experimentIds, currentPage, fetchTraces, pageTokens, filterString, orderByString]);

  const refreshCurrentPage = useCallback(
    (silent = false) => {
      return fetchTraces({
        experimentIds,
        currentPage,
        pageToken: currentPageToken,
        silent,
        filterString,
        orderByString,
      });
    },
    [experimentIds, currentPage, fetchTraces, currentPageToken, filterString, orderByString],
  );

  return {
    traces,
    loading,
    error,
    hasNextPage,
    hasPreviousPage,
    fetchNextPage,
    fetchPrevPage,
    refreshCurrentPage,
    reset,
  };
};

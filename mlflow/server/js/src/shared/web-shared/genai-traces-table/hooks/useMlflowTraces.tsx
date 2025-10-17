import { isNil } from 'lodash';
import { useMemo } from 'react';

import { useIntl } from '@databricks/i18n';
import type { NetworkRequestError } from '@databricks/web-shared/errors';
import { matchPredefinedErrorFromResponse } from '@databricks/web-shared/errors';
import type { QueryClient, UseQueryOptions, UseQueryResult } from '@databricks/web-shared/query-client';
import { useQuery } from '@databricks/web-shared/query-client';

import { useGenAiTraceEvaluationArtifacts } from './useGenAiTraceEvaluationArtifacts';
import {
  EXECUTION_DURATION_COLUMN_ID,
  REQUEST_TIME_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  STATE_COLUMN_ID,
  USER_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  SOURCE_COLUMN_ID,
  useTableColumns,
  CUSTOM_METADATA_COLUMN_ID,
} from './useTableColumns';
import { SourceCellRenderer } from '../cellRenderers/Source/SourceRenderer';
import type { GenAiTraceEvaluationArtifactFile } from '../enum';
import { TracesTableColumnGroup } from '../types';
import type {
  TableFilterOption,
  EvaluationsOverviewTableSort,
  AssessmentFilter,
  RunEvaluationTracesDataEntry,
  TableFilter,
  TraceInfoV3,
  TableFilterOptions,
} from '../types';
import { getAssessmentInfos } from '../utils/AggregationUtils';
import { filterEvaluationResults } from '../utils/EvaluationsFilterUtils';
import {
  shouldEnableUnifiedEvalTab,
  shouldUseRunIdFilterInSearchTraces,
  getMlflowTracesSearchPageSize,
  getEvalTabTotalTracesLimit,
} from '../utils/FeatureUtils';
import { fetchFn, getAjaxUrl } from '../utils/FetchUtils';
import MlflowUtils from '../utils/MlflowUtils';
import { convertTraceInfoV3ToRunEvalEntry, getCustomMetadataKeyFromColumnId } from '../utils/TraceUtils';

interface SearchMlflowTracesRequest {
  locations?: SearchMlflowLocations[];
  filter?: string;
  max_results: number;
  page_token?: string;
  order_by?: string[];
  model_id?: string;
  sql_warehouse_id?: string;
}

interface SearchMlflowLocations {
  type: 'MLFLOW_EXPERIMENT';
  mlflow_experiment: {
    experiment_id: string;
  };
}

export const SEARCH_MLFLOW_TRACES_QUERY_KEY = 'searchMlflowTraces';

export const invalidateMlflowSearchTracesCache = ({ queryClient }: { queryClient: QueryClient }) => {
  queryClient.invalidateQueries({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] });
};

export const useMlflowTracesTableMetadata = ({
  experimentId,
  runUuid,
  timeRange,
  otherRunUuid,
  filterByLoggedModelId,
  loggedModelId,
  sqlWarehouseId,
  disabled,
  networkFilters,
}: {
  experimentId: string;
  runUuid?: string;
  timeRange?: { startTime?: string; endTime?: string };
  otherRunUuid?: string;
  /**
   * Logged model ID to filter offline traces by. Uses trace's request metadata for filtering.
   * To fetch online traces related to a certain logged model, use "loggedModelId" field.
   * N/B: request fields for fetching online and offline traces will be unified in the near future.
   */
  filterByLoggedModelId?: string;
  /**
   * Used to request online traces by logged model ID.
   * If provided, sqlWarehouseId is required in order to query inference tables.
   * N/B: request fields for fetching online and offline traces will be unified in the near future.
   */
  loggedModelId?: string;
  sqlWarehouseId?: string;
  disabled?: boolean;
  networkFilters?: TableFilter[];
}) => {
  const intl = useIntl();
  const filter = createMlflowSearchFilter(runUuid, timeRange, networkFilters, filterByLoggedModelId);
  const {
    data: traces,
    isLoading: isInnerLoading,
    error,
  } = useSearchMlflowTracesInner({
    locations: [{ mlflow_experiment: { experiment_id: experimentId ?? '' }, type: 'MLFLOW_EXPERIMENT' }],
    filter,
    loggedModelId,
    sqlWarehouseId,
    enabled: !disabled,
  });

  const otherFilter = createMlflowSearchFilter(otherRunUuid, timeRange);
  const {
    data: otherTraces,
    isLoading: isOtherInnerLoading,
    error: otherError,
  } = useSearchMlflowTracesInner({
    locations: [{ mlflow_experiment: { experiment_id: experimentId ?? '' }, type: 'MLFLOW_EXPERIMENT' }],
    filter: otherFilter,
    enabled: !disabled && Boolean(otherRunUuid),
    loggedModelId,
    sqlWarehouseId,
  });

  const evaluatedTraces = useMemo(() => {
    if (!traces || isInnerLoading || error || !traces.length) {
      return [];
    }
    return traces.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace));
  }, [traces, isInnerLoading, error]);

  const otherEvaluatedTraces = useMemo(() => {
    const isOtherLoading = isOtherInnerLoading && Boolean(otherRunUuid);
    if (!otherTraces || isOtherLoading || otherError || !otherTraces.length) {
      return [];
    }
    return otherTraces.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace));
  }, [otherTraces, isOtherInnerLoading, otherError, otherRunUuid]);

  const assessmentInfos = useMemo(() => {
    return getAssessmentInfos(intl, evaluatedTraces || [], otherEvaluatedTraces || []);
  }, [intl, evaluatedTraces, otherEvaluatedTraces]);

  const tableFilterOptions = useMemo(() => {
    // Add source options
    const sourceMap = new Map<string, TableFilterOption>();
    traces?.forEach((trace) => {
      const traceMetadata = trace.trace_metadata;
      if (traceMetadata) {
        const sourceName = traceMetadata[MlflowUtils.sourceNameTag];
        if (sourceName && !sourceMap.has(sourceName)) {
          sourceMap.set(sourceName, {
            value: sourceName,
            renderValue: () => <SourceCellRenderer traceInfo={trace} isComparing={false} disableLinks />,
          });
        }
      }
    });

    return {
      source: Array.from(sourceMap.values()).sort((a, b) => a.value.localeCompare(b.value)),
    } as TableFilterOptions;
  }, [traces]);

  const allColumns = useTableColumns(intl, evaluatedTraces || [], assessmentInfos, runUuid, undefined, true);

  return useMemo(() => {
    return {
      assessmentInfos,
      allColumns,
      evaluatedTraces,
      otherEvaluatedTraces,
      totalCount: evaluatedTraces.length,
      isLoading: isInnerLoading && !disabled,
      error,
      isEmpty: evaluatedTraces.length === 0,
      tableFilterOptions,
    };
  }, [
    assessmentInfos,
    allColumns,
    isInnerLoading,
    error,
    evaluatedTraces,
    otherEvaluatedTraces,
    tableFilterOptions,
    disabled,
  ]);
};

const getNetworkAndClientFilters = (
  filters: TableFilter[],
): {
  networkFilters: TableFilter[];
  clientFilters: TableFilter[];
} => {
  return filters.reduce<{
    networkFilters: TableFilter[];
    clientFilters: TableFilter[];
  }>(
    (acc, filter) => {
      // MLflow search api does not support assessment filters, so we need to pass them as client filters
      if (filter.column === TracesTableColumnGroup.ASSESSMENT) {
        acc.clientFilters.push(filter);
      } else {
        acc.networkFilters.push(filter);
      }
      return acc;
    },
    { networkFilters: [], clientFilters: [] },
  );
};

export const useSearchMlflowTraces = ({
  experimentId,
  currentRunDisplayName,
  runUuid,
  timeRange,
  searchQuery,
  filters,
  disabled,
  pageSize,
  limit,
  filterByLoggedModelId,
  tableSort,
  loggedModelId,
  sqlWarehouseId,
}: {
  experimentId: string;
  runUuid?: string | null;
  timeRange?: { startTime?: string; endTime?: string };
  searchQuery?: string;
  filters?: TableFilter[];
  disabled?: boolean;
  pageSize?: number;
  limit?: number;
  // TODO: Remove these once mlflow apis support filtering
  currentRunDisplayName?: string;
  /**
   * Logged model ID to filter offline traces by. Uses trace's request metadata for filtering.
   * To fetch online traces related to a certain logged model, use "loggedModelId" field.
   * N/B: request fields for fetching online and offline traces will be unified in the near future.
   */
  filterByLoggedModelId?: string;
  /**
   * Used to request online traces by logged model ID.
   * If provided, sqlWarehouseId is required in order to query inference tables.
   * N/B: request fields for fetching online and offline traces will be unified in the near future.
   */
  loggedModelId?: string;
  sqlWarehouseId?: string;
  tableSort?: EvaluationsOverviewTableSort;
}): {
  data: TraceInfoV3[] | undefined;
  isLoading: boolean;
  isFetching: boolean;
  error?: NetworkRequestError;
  refetchMlflowTraces?: UseQueryResult<TraceInfoV3[], NetworkRequestError>['refetch'];
} => {
  const { networkFilters, clientFilters } = getNetworkAndClientFilters(filters || []);

  const filter = createMlflowSearchFilter(runUuid, timeRange, networkFilters, filterByLoggedModelId);
  const orderBy = createMlflowSearchOrderBy(tableSort);

  const {
    data: traces,
    isLoading: isInnerLoading,
    isFetching: isInnerFetching,
    error,
    refetch: refetchMlflowTraces,
  } = useSearchMlflowTracesInner({
    locations: [{ mlflow_experiment: { experiment_id: experimentId ?? '' }, type: 'MLFLOW_EXPERIMENT' }],
    filter,
    enabled: !disabled,
    pageSize,
    limit,
    orderBy,
    loggedModelId,
    sqlWarehouseId,
  });

  // TODO: Remove this once mlflow apis support filtering
  const evalTraceComparisonEntries = useMemo(() => {
    if (!traces) {
      return undefined;
    }

    return traces.map((trace) => {
      return {
        currentRunValue: convertTraceInfoV3ToRunEvalEntry(trace),
        otherRunValue: undefined,
      };
    });
  }, [traces]);

  // TODO: Remove this once mlflow apis support filtering
  const filteredTraces: TraceInfoV3[] | undefined = useMemo(() => {
    if (!evalTraceComparisonEntries) return undefined;

    if (searchQuery === '' && clientFilters?.length === 0) {
      return evalTraceComparisonEntries.reduce<TraceInfoV3[]>((acc, entry) => {
        if (entry.currentRunValue?.traceInfo) {
          acc.push(entry.currentRunValue.traceInfo);
        }
        return acc;
      }, []);
    }

    const assessmentFilters: AssessmentFilter[] = clientFilters.map((filter) => {
      return {
        assessmentName: filter.key || '',
        filterValue: filter.value,
        run: currentRunDisplayName || '',
      };
    });

    const res = filterEvaluationResults(
      evalTraceComparisonEntries,
      assessmentFilters || [],
      searchQuery,
      currentRunDisplayName,
      undefined,
    ).reduce<TraceInfoV3[]>((acc, entry) => {
      if (entry.currentRunValue?.traceInfo) {
        acc.push(entry.currentRunValue.traceInfo);
      }
      return acc;
    }, []);

    return res;
  }, [evalTraceComparisonEntries, clientFilters, searchQuery, currentRunDisplayName]);

  if (disabled) {
    return {
      data: [],
      isLoading: false,
      isFetching: false,
    };
  }

  return {
    data: filteredTraces,
    isLoading: isInnerLoading,
    isFetching: isInnerFetching,
    error: error || undefined,
    refetchMlflowTraces,
  };
};

/**
 * Fetches all mlflow traces for a given location/filter in a synchronous loop.
 * The results of all the traces are cached under a single key.
 * TODO: De-dup with useSearchMlflowTraces defined in webapp/web/js/genai
 */
const useSearchMlflowTracesInner = ({
  locations,
  filter,
  pageSize: pageSizeProp,
  limit: limitProp,
  orderBy,
  loggedModelId,
  sqlWarehouseId,
  enabled,
  ...rest
}: {
  locations?: SearchMlflowLocations[];
  filter?: string;
  pageSize?: number;
  limit?: number;
  orderBy?: string[];
  loggedModelId?: string;
  sqlWarehouseId?: string;
} & Omit<UseQueryOptions<TraceInfoV3[], NetworkRequestError>, 'queryFn'>) => {
  return useQuery<TraceInfoV3[], NetworkRequestError>({
    staleTime: Infinity,
    cacheTime: Infinity,
    enabled,
    queryKey: [
      SEARCH_MLFLOW_TRACES_QUERY_KEY,
      {
        experimentIds: (locations || []).map((x) => x.mlflow_experiment?.experiment_id),
        filter,
        orderBy,
        loggedModelId,
        sqlWarehouseId,
      },
    ],
    queryFn: async ({ signal }) => {
      let allTraces: TraceInfoV3[] = [];
      let pageToken: string | undefined = undefined;

      const pageSize = pageSizeProp || getMlflowTracesSearchPageSize();
      const tracesLimit = limitProp || getEvalTabTotalTracesLimit();

      while (allTraces.length < tracesLimit) {
        const payload: SearchMlflowTracesRequest = {
          locations,
          filter,
          max_results: pageSize,
          order_by: orderBy,
        };
        if (loggedModelId && sqlWarehouseId) {
          payload.model_id = loggedModelId;
          payload.sql_warehouse_id = sqlWarehouseId;
        }
        if (pageToken) {
          payload.page_token = pageToken;
        }
        const queryResponse = await fetchFn(getAjaxUrl('ajax-api/3.0/mlflow/traces/search'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
          signal,
        });
        if (!queryResponse.ok) throw matchPredefinedErrorFromResponse(queryResponse);
        const json = (await queryResponse.json()) as { traces: TraceInfoV3[]; next_page_token?: string };
        const traces = json.traces;
        if (!isNil(traces)) {
          allTraces = allTraces.concat(traces);
        }

        // If there's no next page, break out of the loop.
        pageToken = json.next_page_token;
        if (!pageToken) break;
      }
      return allTraces;
    },
    ...rest,
  });
};

/**
 * Builds evaluation trace entries from search results if no artifact data is present.
 * Falls back to provided artifactData when available or on error.
 *
 * @param artifactData - Existing evaluation trace entries from artifacts.
 * @param searchRes - Query result containing TraceInfoV3 entries.
 * @returns A list of RunEvaluationTracesDataEntry either from artifactData or initialized from search results. Also
 * returns shouldUseTraceV3 indicating if we are using artifacts or trace v3 results from search
 */
const buildTracesFromSearchAndArtifacts = (
  artifactData: RunEvaluationTracesDataEntry[],
  searchRes: UseQueryResult<TraceInfoV3[], NetworkRequestError>,
): {
  data: RunEvaluationTracesDataEntry[];
  shouldUseTraceV3: boolean;
  error?: NetworkRequestError;
} => {
  const { data: searchData, error } = searchRes;

  if (artifactData.length > 0 || error || !searchData || searchData.length === 0) {
    return { data: artifactData, shouldUseTraceV3: false, error: error || undefined };
  }

  // We want to start using information from TraceInfoV3 downstream rather
  // than RunEvaluationTracesDataEntry, so fill in all properties as empty
  // except for traceInfo.
  return {
    data: searchData
      .filter((trace): trace is TraceInfoV3 => trace !== null && trace !== undefined)
      .map((trace) => {
        return {
          evaluationId: '',
          requestId: '',
          inputs: {},
          inputsId: '',
          outputs: {},
          targets: {},
          overallAssessments: [],
          responseAssessmentsByName: {},
          metrics: {},
          traceInfo: trace,
        };
      }),
    shouldUseTraceV3: true,
  };
};

const createMlflowSearchFilter = (
  runUuid: string | null | undefined,
  timeRange?: { startTime?: string; endTime?: string } | null,
  networkFilters?: TableFilter[],
  loggedModelId?: string,
) => {
  const filter: string[] = [];
  const useRunIdInSearchTraces = shouldUseRunIdFilterInSearchTraces();
  if (runUuid) {
    if (useRunIdInSearchTraces) {
      filter.push(`attributes.run_id = '${runUuid}'`);
    } else {
      filter.push(`request_metadata."mlflow.sourceRun" = '${runUuid}'`);
    }
  }
  if (timeRange) {
    const timestampField = 'attributes.timestamp_ms';
    if (timeRange.startTime) {
      filter.push(`${timestampField} > ${timeRange.startTime}`);
    }
    if (timeRange.endTime) {
      filter.push(`${timestampField} < ${timeRange.endTime}`);
    }
  }
  if (loggedModelId) {
    filter.push(`request_metadata."mlflow.modelId" = '${loggedModelId}'`);
  }
  if (networkFilters) {
    networkFilters.forEach((networkFilter) => {
      switch (networkFilter.column) {
        case TracesTableColumnGroup.TAG:
          if (networkFilter.key) {
            const tagField = 'tags';
            // Use backticks for field names with special characters
            const fieldName =
              networkFilter.key.includes('.') || networkFilter.key.includes(' ')
                ? `${tagField}.\`${networkFilter.key}\``
                : `${tagField}.${networkFilter.key}`;
            filter.push(`${fieldName} ${networkFilter.operator} '${networkFilter.value}'`);
          }
          break;
        case EXECUTION_DURATION_COLUMN_ID:
          const executionField = 'attributes.execution_time_ms';
          filter.push(`${executionField} ${networkFilter.operator} ${networkFilter.value}`);
          break;
        case STATE_COLUMN_ID:
          const statusField = 'attributes.status';
          filter.push(`${statusField} = '${networkFilter.value}'`);
          break;
        case USER_COLUMN_ID:
          filter.push(`request_metadata."mlflow.trace.user" = '${networkFilter.value}'`);
          break;
        case RUN_NAME_COLUMN_ID:
          if (useRunIdInSearchTraces) {
            filter.push(`attributes.run_id = '${networkFilter.value}'`);
          } else {
            filter.push(`request_metadata."mlflow.sourceRun" = '${networkFilter.value}'`);
          }
          break;
        case LOGGED_MODEL_COLUMN_ID:
          filter.push(`request_metadata."mlflow.modelId" = '${networkFilter.value}'`);
          break;
        case TRACE_NAME_COLUMN_ID:
          filter.push(`attributes.name ${networkFilter.operator} '${networkFilter.value}'`);
          break;
        case SOURCE_COLUMN_ID:
          filter.push(`request_metadata."mlflow.source.name" ${networkFilter.operator} '${networkFilter.value}'`);
          break;
        default:
          if (networkFilter.column.startsWith(CUSTOM_METADATA_COLUMN_ID)) {
            filter.push(
              `request_metadata.${getCustomMetadataKeyFromColumnId(networkFilter.column)} ${networkFilter.operator} '${
                networkFilter.value
              }'`,
            );
          }
          break;
      }
    });
  }

  if (filter.length > 0) {
    return filter.join(' AND ');
  }
  return undefined;
};

const createMlflowSearchOrderBy = (tableSort?: EvaluationsOverviewTableSort): string[] | undefined => {
  if (!tableSort) {
    return undefined;
  }

  // Currently the server only supports sorting by execution time and request time. Should add
  // more columns as they are supported by the server.
  switch (tableSort.key) {
    case EXECUTION_DURATION_COLUMN_ID:
      return [`execution_time ${tableSort.asc ? 'ASC' : 'DESC'}`];
    case REQUEST_TIME_COLUMN_ID:
      return [`timestamp ${tableSort.asc ? 'ASC' : 'DESC'}`];
    default:
      return [];
  }
};

/**
 * Fetches all mlflow traces for a given experiment id and run id.
 * It returns the traces from artifacts storage if they exist and uses search API otherwise.
 *
 * @param experimentId - The experiment id to fetch traces for.
 * @param runUuid - The run id to fetch traces for. If not provided, it will fetch traces for the entire experiment.
 * @param traceTablesLoggedInRun - The trace tables logged in run to fetch traces for. Only used for fetching from artifacts storage. If not provided, it will fetch from all artifacts
 * @param disabled - Whether to disable the traces call.
 * @param timeRange - The time range to fetch traces for. Start/End time should be in milliseconds since epoch.
 */
export const useMlflowTraces = (
  experimentId?: string | null,
  runUuid?: string | null,
  traceTablesLoggedInRun?: GenAiTraceEvaluationArtifactFile[],
  disabled?: boolean,
  timeRange?: {
    startTime: string | undefined;
    endTime: string | undefined;
  },
): {
  data: RunEvaluationTracesDataEntry[];
  isLoading: boolean;
  refetchMlflowTraces?: () => Promise<any>;
  shouldUseTraceV3: boolean;
  error?: NetworkRequestError;
} => {
  const isUnifiedEvalTabEnabled = shouldEnableUnifiedEvalTab();

  const isExperimentIdValid = Boolean(experimentId && experimentId.length > 0);
  const isArtifactCallEnabled = Boolean(!disabled && runUuid);
  const isTracesCallEnabled = isExperimentIdValid && Boolean(isUnifiedEvalTabEnabled && !disabled);

  const { data: artifactData, isLoading: isArtifactLoading } = useGenAiTraceEvaluationArtifacts(
    {
      runUuid: runUuid || '',
      ...{ artifacts: traceTablesLoggedInRun ? traceTablesLoggedInRun : undefined },
    },
    { disabled: !isArtifactCallEnabled },
  );

  const filter = createMlflowSearchFilter(runUuid, timeRange);

  const searchRes = useSearchMlflowTracesInner({
    locations: [{ mlflow_experiment: { experiment_id: experimentId ?? '' }, type: 'MLFLOW_EXPERIMENT' }],
    filter,
    enabled: isTracesCallEnabled,
  });

  if (disabled) {
    return {
      data: [],
      isLoading: false,
      shouldUseTraceV3: false,
    };
  }

  if (!isUnifiedEvalTabEnabled) {
    return {
      data: artifactData || [],
      isLoading: isArtifactLoading,
      shouldUseTraceV3: false,
    };
  }

  return {
    ...buildTracesFromSearchAndArtifacts(artifactData || [], searchRes),
    isLoading: isArtifactLoading || (searchRes.isLoading && isTracesCallEnabled),
    refetchMlflowTraces: searchRes.refetch,
  };
};

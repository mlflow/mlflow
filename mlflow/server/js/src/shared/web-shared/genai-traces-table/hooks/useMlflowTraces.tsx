import { isNil } from 'lodash';
import { useMemo } from 'react';

import { useIntl } from '@databricks/i18n';
import type { NetworkRequestError } from '@databricks/web-shared/errors';
import { matchPredefinedErrorFromResponse } from '@databricks/web-shared/errors';
import type { QueryClient, UseQueryOptions, UseQueryResult } from '@databricks/web-shared/query-client';
import { useQuery } from '@databricks/web-shared/query-client';

import {
  EXECUTION_DURATION_COLUMN_ID,
  REQUEST_TIME_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  STATE_COLUMN_ID,
  USER_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  LINKED_PROMPTS_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  SOURCE_COLUMN_ID,
  useTableColumns,
  CUSTOM_METADATA_COLUMN_ID,
  SPAN_NAME_COLUMN_ID,
  SPAN_TYPE_COLUMN_ID,
  SPAN_CONTENT_COLUMN_ID,
  INPUTS_COLUMN_ID,
  RESPONSE_COLUMN_ID,
} from './useTableColumns';
import {
  TracesServiceV4,
  type ModelTraceInfoV3,
  type ModelTraceLocationMlflowExperiment,
  type ModelTraceLocationUcSchema,
} from '../../model-trace-explorer';
import { SourceCellRenderer } from '../cellRenderers/Source/SourceRenderer';
import type {
  TableFilterOption,
  EvaluationsOverviewTableSort,
  AssessmentFilter,
  RunEvaluationTracesDataEntry,
  TableFilter,
  TableFilterOptions,
} from '../types';
import { FilterOperator, HiddenFilterOperator, TracesTableColumnGroup, TracesTableColumnType } from '../types';
import { ERROR_KEY, getAssessmentInfos } from '../utils/AggregationUtils';
import { filterEvaluationResults } from '../utils/EvaluationsFilterUtils';
import { getMlflowTracesSearchPageSize, getEvalTabTotalTracesLimit, shouldUseTracesV4API } from '../utils/FeatureUtils';
import { fetchFn, getAjaxUrl } from '../utils/FetchUtils';
import MlflowUtils from '../utils/MlflowUtils';
import {
  convertTraceInfoV3ToRunEvalEntry,
  filterTracesByAssessmentSourceRunId,
  getCustomMetadataKeyFromColumnId,
} from '../utils/TraceUtils';

interface SearchMlflowTracesRequest {
  locations?: (ModelTraceLocationMlflowExperiment | ModelTraceLocationUcSchema)[];
  filter?: string;
  max_results: number;
  page_token?: string;
  order_by?: string[];
  model_id?: string;
  sql_warehouse_id?: string;
}

export const SEARCH_MLFLOW_TRACES_QUERY_KEY = 'searchMlflowTraces';

export const invalidateMlflowSearchTracesCache = ({ queryClient }: { queryClient: QueryClient }) => {
  queryClient.invalidateQueries({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] });
};

const defaultTableSort: EvaluationsOverviewTableSort = {
  asc: false,
  key: REQUEST_TIME_COLUMN_ID,
  type: TracesTableColumnType.TRACE_INFO,
};

export const useMlflowTracesTableMetadata = ({
  locations,
  runUuid,
  timeRange,
  otherRunUuid,
  filterByLoggedModelId,
  loggedModelId,
  sqlWarehouseId,
  disabled,
  networkFilters,
  filterByAssessmentSourceRun = false,
  evalStartTime,
}: {
  locations: (ModelTraceLocationMlflowExperiment | ModelTraceLocationUcSchema)[];
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
  /**
   * If true, filters traces by assessment source run ID. This is used in the eval tab
   * to only show assessments that were created by the current evaluation run.
   * Defaults to false for other tabs (traces, labeling, etc.).
   */
  filterByAssessmentSourceRun?: boolean;
  /**
   * Optional timestamp (ms) when evaluation started. If provided during an active
   * evaluation, traces created after this time are filtered out.
   */
  evalStartTime?: number | null;
}) => {
  const intl = useIntl();
  const filter = createMlflowSearchFilter(runUuid, timeRange, networkFilters, filterByLoggedModelId);
  const usingV4APIs = locations?.some((location) => location.type === 'UC_SCHEMA') && shouldUseTracesV4API();

  const orderBy = createMlflowSearchOrderBy(defaultTableSort);

  const {
    data: traces,
    isLoading: isInnerLoading,
    error,
  } = useSearchMlflowTracesInner({
    locations,
    filter,
    loggedModelId,
    sqlWarehouseId,
    enabled: !disabled,
    orderBy,
  });
  const filteredTraces = useMemo(
    () => (filterByAssessmentSourceRun ? filterTracesByAssessmentSourceRunId(traces, runUuid, evalStartTime) : traces),
    [traces, runUuid, filterByAssessmentSourceRun, evalStartTime],
  );

  const otherFilter = createMlflowSearchFilter(otherRunUuid, timeRange);
  const {
    data: otherTraces,
    isLoading: isOtherInnerLoading,
    error: otherError,
  } = useSearchMlflowTracesInner({
    locations,
    filter: otherFilter,
    enabled: !disabled && Boolean(otherRunUuid),
    loggedModelId,
    sqlWarehouseId,
    orderBy,
  });

  const filteredOtherTraces = useMemo(
    () => (filterByAssessmentSourceRun ? filterTracesByAssessmentSourceRunId(otherTraces, otherRunUuid) : otherTraces),
    [otherTraces, otherRunUuid, filterByAssessmentSourceRun],
  );

  const evaluatedTraces = useMemo(() => {
    if (!filteredTraces || isInnerLoading || error || !filteredTraces.length) {
      return [];
    }
    return filteredTraces.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace));
  }, [filteredTraces, isInnerLoading, error]);

  const otherEvaluatedTraces = useMemo(() => {
    const isOtherLoading = isOtherInnerLoading && Boolean(otherRunUuid);
    if (!filteredOtherTraces || isOtherLoading || otherError || !filteredOtherTraces.length) {
      return [];
    }
    return filteredOtherTraces.map((trace) => convertTraceInfoV3ToRunEvalEntry(trace));
  }, [filteredOtherTraces, isOtherInnerLoading, otherError, otherRunUuid]);

  const assessmentInfos = useMemo(() => {
    return getAssessmentInfos(intl, evaluatedTraces || [], otherEvaluatedTraces || []);
  }, [intl, evaluatedTraces, otherEvaluatedTraces]);

  const tableFilterOptions = useMemo(() => {
    // Add source options
    const sourceMap = new Map<string, TableFilterOption>();
    const promptMap = new Map<string, TableFilterOption>();

    filteredTraces?.forEach((trace) => {
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

      // Extract linked prompts from trace tags. Fetch data from backend if users
      // want to filter by prompt not included in the current traces.
      const tags = trace.tags;
      if (tags && tags['mlflow.linkedPrompts']) {
        try {
          const linkedPrompts = JSON.parse(tags['mlflow.linkedPrompts']);
          if (Array.isArray(linkedPrompts)) {
            linkedPrompts.forEach((prompt: { name: string; version: string }) => {
              const promptValue = `${prompt.name}/${prompt.version}`;
              if (!promptMap.has(promptValue)) {
                promptMap.set(promptValue, {
                  value: promptValue,
                  renderValue: () => promptValue,
                });
              }
            });
          }
        } catch (e) {
          // Ignore invalid JSON
        }
      }
    });

    return {
      source: Array.from(sourceMap.values()).sort((a, b) => a.value.localeCompare(b.value)),
      prompt: Array.from(promptMap.values()).sort((a, b) => a.value.localeCompare(b.value)),
    } as TableFilterOptions;
  }, [filteredTraces]);

  const allColumns = useTableColumns(intl, evaluatedTraces || [], assessmentInfos, runUuid, undefined, true);

  return useMemo(() => {
    return {
      assessmentInfos,
      allColumns,
      totalCount: evaluatedTraces.length,
      evaluatedTraces,
      otherEvaluatedTraces,
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
    tableFilterOptions,
    disabled,
    otherEvaluatedTraces,
  ]);
};

const getNetworkAndClientFilters = (
  filters: TableFilter[],
  assessmentsFilteredOnClientSide = true,
): {
  networkFilters: TableFilter[];
  clientFilters: TableFilter[];
} => {
  return filters.reduce<{
    networkFilters: TableFilter[];
    clientFilters: TableFilter[];
  }>(
    (acc, filter) => {
      // Assessment filters with undefined or 'Error' value must always be filtered client-side
      // because the backend cannot query for absence of an assessment or error state.
      // Note: filter.value is already converted from string 'undefined' to actual undefined by useFilters
      const isClientOnlyAssessmentFilter =
        filter.column === TracesTableColumnGroup.ASSESSMENT &&
        (filter.value === undefined || filter.value === ERROR_KEY);

      if (isClientOnlyAssessmentFilter) {
        acc.clientFilters.push(filter);
      } else if (filter.column === TracesTableColumnGroup.ASSESSMENT && assessmentsFilteredOnClientSide) {
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
  locations,
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
  filterByAssessmentSourceRun = false,
  evalStartTime,
}: {
  locations: (ModelTraceLocationMlflowExperiment | ModelTraceLocationUcSchema)[];
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
  /**
   * If true, filters traces by assessment source run ID. This is used in the eval tab
   * to only show assessments that were created by the current evaluation run.
   * Defaults to false for other tabs (traces, labeling, etc.).
   */
  filterByAssessmentSourceRun?: boolean;
  /**
   * Optional timestamp (ms) when evaluation started. If provided during an active
   * evaluation, traces created after this time are filtered out.
   */
  evalStartTime?: number | null;
}): {
  data: ModelTraceInfoV3[] | undefined;
  isLoading: boolean;
  isFetching: boolean;
  error?: NetworkRequestError;
  refetchMlflowTraces?: UseQueryResult<ModelTraceInfoV3[], NetworkRequestError>['refetch'];
} => {
  // Client-side filtering is always disabled in OSS MLflow. It is only used in Databricks.
  const useClientSideFiltering = false;

  const { networkFilters, clientFilters } = useMemo(
    () => getNetworkAndClientFilters(filters || [], useClientSideFiltering),
    [filters, useClientSideFiltering],
  );

  const filter = createMlflowSearchFilter(
    runUuid,
    timeRange,
    networkFilters,
    filterByLoggedModelId,
    useClientSideFiltering ? undefined : searchQuery,
  );
  const orderBy = createMlflowSearchOrderBy(tableSort);

  const {
    data: traces,
    isLoading: isInnerLoading,
    isFetching: isInnerFetching,
    error,
    refetch: refetchMlflowTraces,
  } = useSearchMlflowTracesInner({
    locations,
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

  const filteredTraces: ModelTraceInfoV3[] | undefined = useMemo(() => {
    if (!evalTraceComparisonEntries) return undefined;

    const hasClientFilters = clientFilters && clientFilters.length > 0;
    // Only apply client-side search filtering when useClientSideFiltering is true.
    // When false, searchQuery is already applied server-side via createMlflowSearchFilter.
    const clientSearchQuery = useClientSideFiltering ? searchQuery : undefined;
    const hasClientSearchQuery = clientSearchQuery && clientSearchQuery !== '';

    // Skip filtering if there are no client filters to apply
    if (!hasClientFilters && !hasClientSearchQuery) {
      return evalTraceComparisonEntries.reduce<ModelTraceInfoV3[]>((acc, entry) => {
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
      clientSearchQuery,
      currentRunDisplayName,
      undefined,
    ).reduce<ModelTraceInfoV3[]>((acc, entry) => {
      if (entry.currentRunValue?.traceInfo) {
        acc.push(entry.currentRunValue.traceInfo);
      }
      return acc;
    }, []);

    return res;
  }, [evalTraceComparisonEntries, clientFilters, searchQuery, currentRunDisplayName, useClientSideFiltering]);

  const tracesFilteredBySourceRun = useMemo(
    () =>
      filterByAssessmentSourceRun
        ? filterTracesByAssessmentSourceRunId(filteredTraces, runUuid, evalStartTime)
        : filteredTraces,
    [filteredTraces, runUuid, filterByAssessmentSourceRun, evalStartTime],
  );

  if (disabled) {
    return {
      data: [],
      isLoading: false,
      isFetching: false,
    };
  }

  return {
    data: tracesFilteredBySourceRun,
    isLoading: isInnerLoading,
    isFetching: isInnerFetching,
    error: error || undefined,
    refetchMlflowTraces,
  };
};

/**
 * Query function for searching MLflow traces.
 * Fetches all traces for given locations/filters, paginating through results.
 */
export const searchMlflowTracesQueryFn = async ({
  signal,
  locations,
  filter,
  pageSize: pageSizeProp,
  limit: limitProp,
  orderBy,
  loggedModelId,
  sqlWarehouseId,
}: {
  signal?: AbortSignal;
  locations?: (ModelTraceLocationMlflowExperiment | ModelTraceLocationUcSchema)[];
  filter?: string;
  pageSize?: number;
  limit?: number;
  orderBy?: string[];
  loggedModelId?: string;
  sqlWarehouseId?: string;
}): Promise<ModelTraceInfoV3[]> => {
  const usingV4APIs = locations?.some((location) => location.type === 'UC_SCHEMA') && shouldUseTracesV4API();

  if (usingV4APIs) {
    return TracesServiceV4.searchTracesV4({
      signal,
      orderBy,
      locations,
      filter,
      pageSize: pageSizeProp,
    });
  }
  let allTraces: ModelTraceInfoV3[] = [];
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
    const json = (await queryResponse.json()) as { traces: ModelTraceInfoV3[]; next_page_token?: string };
    const traces = json.traces;
    if (!isNil(traces)) {
      allTraces = allTraces.concat(traces);
    }

    // If there's no next page, break out of the loop.
    pageToken = json.next_page_token;
    if (!pageToken) break;
  }
  return allTraces;
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
  locations?: (ModelTraceLocationMlflowExperiment | ModelTraceLocationUcSchema)[];
  filter?: string;
  pageSize?: number;
  limit?: number;
  orderBy?: string[];
  loggedModelId?: string;
  sqlWarehouseId?: string;
} & Omit<UseQueryOptions<ModelTraceInfoV3[], NetworkRequestError>, 'queryFn'>) => {
  const usingV4APIs = locations?.some((location) => location.type === 'UC_SCHEMA') && shouldUseTracesV4API();

  // In V4 API, we use server-side for all filters so we can keep previous data to smooth out UX and use default cache values.
  // In previous APIs, we relied on client-side filtering so we want to cache indefinitely.
  const queryCacheConfig = useMemo(
    () =>
      usingV4APIs
        ? {
            keepPreviousData: true,
            refetchOnWindowFocus: false,
          }
        : {
            staleTime: Infinity,
            cacheTime: Infinity,
          },
    [usingV4APIs],
  );

  return useQuery<ModelTraceInfoV3[], NetworkRequestError>({
    ...queryCacheConfig,
    enabled,
    queryKey: [
      SEARCH_MLFLOW_TRACES_QUERY_KEY,
      {
        locations,
        filter,
        orderBy,
        loggedModelId,
        sqlWarehouseId,
        pageSize: pageSizeProp,
      },
    ],
    queryFn: ({ signal }) =>
      searchMlflowTracesQueryFn({
        signal,
        locations,
        filter,
        pageSize: pageSizeProp,
        limit: limitProp,
        orderBy,
        loggedModelId,
        sqlWarehouseId,
      }),
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
  searchRes: UseQueryResult<ModelTraceInfoV3[], NetworkRequestError>,
  runUuid?: string | null,
): {
  data: RunEvaluationTracesDataEntry[];
  shouldUseTraceV3: boolean;
  error?: NetworkRequestError;
} => {
  const { data: searchData, error } = searchRes;
  const filteredSearchData = filterTracesByAssessmentSourceRunId(searchData, runUuid);

  if (artifactData.length > 0 || error || !filteredSearchData || filteredSearchData.length === 0) {
    return { data: artifactData, shouldUseTraceV3: false, error: error || undefined };
  }

  // We want to start using information from TraceInfoV3 downstream rather
  // than RunEvaluationTracesDataEntry, so fill in all properties as empty
  // except for traceInfo.
  return {
    data: filteredSearchData
      .filter((trace): trace is ModelTraceInfoV3 => trace !== null && trace !== undefined)
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

export const createMlflowSearchFilter = (
  runUuid: string | null | undefined,
  timeRange?: { startTime?: string; endTime?: string } | null,
  networkFilters?: TableFilter[],
  loggedModelId?: string,
  searchQuery?: string,
) => {
  const filter: string[] = [];
  if (runUuid) {
    filter.push(`attributes.run_id = '${runUuid}'`);
  }
  if (searchQuery) {
    filter.push(`span.attributes.\`mlflow.spanInputs\` ILIKE '%${searchQuery}%'`);
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
          filter.push(`attributes.run_id = '${networkFilter.value}'`);
          break;
        case LOGGED_MODEL_COLUMN_ID:
          filter.push(`request_metadata."mlflow.modelId" = '${networkFilter.value}'`);
          break;
        // Only available in OSS
        case LINKED_PROMPTS_COLUMN_ID:
          filter.push(`prompt = '${networkFilter.value}'`);
          break;
        case TRACE_NAME_COLUMN_ID:
          filter.push(`attributes.name ${networkFilter.operator} '${networkFilter.value}'`);
          break;
        case SOURCE_COLUMN_ID:
          filter.push(`request_metadata."mlflow.source.name" ${networkFilter.operator} '${networkFilter.value}'`);
          break;
        case TracesTableColumnGroup.ASSESSMENT:
          // Skip 'undefined' values - these must be filtered client-side since they represent
          // absence of an assessment, which cannot be queried on the backend
          if (networkFilter.value !== 'undefined') {
            filter.push(`feedback.\`${networkFilter.key}\` ${networkFilter.operator} '${networkFilter.value}'`);
          }
          break;
        case TracesTableColumnGroup.EXPECTATION:
          filter.push(`expectation.\`${networkFilter.key}\` ${networkFilter.operator} '${networkFilter.value}'`);
          break;
        case SPAN_NAME_COLUMN_ID:
          if (networkFilter.operator === '=') {
            // Use ILIKE instead of = for case-insensitive matching (better UX for span name filtering)
            filter.push(`span.name ILIKE '${networkFilter.value}'`);
          } else if (networkFilter.operator === 'CONTAINS') {
            filter.push(`span.name ILIKE '%${networkFilter.value}%'`);
          } else {
            filter.push(`span.name ${networkFilter.operator} '${networkFilter.value}'`);
          }
          break;
        case INPUTS_COLUMN_ID:
        case RESPONSE_COLUMN_ID:
          filter.push(`${networkFilter.column} ${networkFilter.operator} '${networkFilter.value}'`);
          break;
        case SPAN_TYPE_COLUMN_ID:
          if (networkFilter.operator === '=') {
            // Use ILIKE instead of = for case-insensitive matching (better UX for span type filtering)
            filter.push(`span.type ILIKE '${networkFilter.value}'`);
          } else if (networkFilter.operator === 'CONTAINS') {
            filter.push(`span.type ILIKE '%${networkFilter.value}%'`);
          } else {
            filter.push(`span.type ${networkFilter.operator} '${networkFilter.value}'`);
          }
          break;
        case SPAN_CONTENT_COLUMN_ID:
          if (networkFilter.operator === 'CONTAINS') {
            filter.push(`span.content ILIKE '%${networkFilter.value}%'`);
          }
          break;
        default:
          if (networkFilter.column.startsWith(CUSTOM_METADATA_COLUMN_ID)) {
            const columnKey = `request_metadata.${getCustomMetadataKeyFromColumnId(networkFilter.column)}`;
            if (networkFilter.operator === HiddenFilterOperator.IS_NOT_NULL) {
              filter.push(`${columnKey} IS NOT NULL`);
            } else if (networkFilter.operator === FilterOperator.CONTAINS) {
              filter.push(`${columnKey} ILIKE '%${networkFilter.value}%'`);
            } else {
              filter.push(`${columnKey} ${networkFilter.operator} '${networkFilter.value}'`);
            }
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

import { isNil } from 'lodash';
import { useMemo } from 'react';

import { useIntl } from '@databricks/i18n';
import type { NetworkRequestError } from '../../errors/PredefinedErrors';
import type { QueryClient } from '../../query-client/queryClient';
import { useQuery, useInfiniteQuery } from '../../query-client/queryClient';
import {
  isV4TraceId,
  parseV4TraceId,
  parseTraceV4SerializedLocation,
  createTraceV4SerializedLocation,
} from '../../model-trace-explorer/ModelTraceExplorer.utils';

import {
  EXECUTION_DURATION_COLUMN_ID,
  REQUEST_TIME_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  SESSION_COLUMN_ID,
  STATE_COLUMN_ID,
  USER_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  SOURCE_COLUMN_ID,
  useTableColumns,
  LINKED_PROMPTS_COLUMN_ID,
  CUSTOM_METADATA_COLUMN_ID,
  SPAN_NAME_COLUMN_ID,
  SPAN_TYPE_COLUMN_ID,
  SPAN_STATUS_COLUMN_ID,
  SPAN_CONTENT_COLUMN_ID,
  INPUTS_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  ISSUE_ID_COLUMN_ID,
} from './useTableColumns';
import { TracesServiceV4, fetchTraceInfoV3 } from '../../model-trace-explorer/api';
import type { ModelTraceInfoV3, ModelTraceSearchLocation } from '../../model-trace-explorer/ModelTrace.types';
import { SourceCellRenderer } from '../cellRenderers/Source/SourceRenderer';
import type {
  TableFilterOption,
  EvaluationsOverviewTableSort,
  AssessmentFilter,
  TableFilter,
  TableFilterOptions,
} from '../types';
import {
  FilterOperator,
  HiddenFilterOperator,
  TracesTableColumnGroup,
  TracesTableColumnType,
  isNullOperator,
} from '../types';
import { ERROR_KEY, getAssessmentInfos } from '../utils/AggregationUtils';
import { filterEvaluationResults } from '../utils/EvaluationsFilterUtils';
import {
  getMlflowTracesSearchPageSize,
  getEvalTabTotalTracesLimit,
  shouldUseTracesV4API,
  shouldUseLongRunningTracesAPI,
  shouldUseInfinitePaginatedTraces,
} from '../utils/FeatureUtils';
import { fetchAPI, getAjaxUrl } from '../utils/FetchUtils';
import MlflowUtils from '../utils/MlflowUtils';
import {
  convertTraceInfoV3ToRunEvalEntry,
  filterTracesByAssessmentSourceRunId,
  getCustomMetadataKeyFromColumnId,
} from '../utils/TraceUtils';
import { isV4TraceLocation } from '../utils/TraceLocationUtils';

interface SearchMlflowTracesRequest {
  locations?: ModelTraceSearchLocation[];
  filter?: string;
  max_results: number;
  page_token?: string;
  order_by?: string[];
  model_id?: string;
  sql_warehouse_id?: string;
}

export const SEARCH_MLFLOW_TRACES_QUERY_KEY = 'searchMlflowTraces';
const TRACE_ID_LOOKUP_QUERY_KEY = 'traceIdLookup';

export const invalidateMlflowSearchTracesCache = ({ queryClient }: { queryClient: QueryClient }) => {
  queryClient.invalidateQueries({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] });
};

/**
 * Hex string pattern: 32-char hex strings (common backend trace ID format).
 */
const HEX_TRACE_ID_PATTERN = /^[0-9a-fA-F]{32}$/;

/**
 * Detects whether a search query looks like a trace ID.
 * Supports:
 *   - Full V4 trace ID: trace:/catalog.schema/abc123...
 *   - Backend trace ID: 32-character hex string (e.g. 11301f0bdf2dfa5a762a4bac74b45db1)
 */
export const extractTraceIdFromSearchQuery = (
  searchQuery: string,
): { backendTraceId: string; traceLocation?: string } | undefined => {
  const trimmed = searchQuery.trim();

  // Check for V4 full trace ID format: trace:/location/traceId
  if (isV4TraceId(trimmed)) {
    const parsed = parseV4TraceId(trimmed);
    if (parsed?.trace_id && parsed?.trace_location) {
      return { backendTraceId: parsed.trace_id, traceLocation: parsed.trace_location };
    }
    return undefined;
  }

  // Check for plain backend trace ID (32-char hex string)
  if (HEX_TRACE_ID_PATTERN.test(trimmed)) {
    return { backendTraceId: trimmed };
  }

  return undefined;
};

/**
 * Hook that looks up a single trace by trace ID when the search query appears to be a trace ID.
 * Uses get_trace / batch get API to fetch the trace directly, since trace_id is not a
 * searchable field in the search traces API.
 */
const useTraceIdLookup = ({
  searchQuery,
  locations,
  sqlWarehouseId,
  enabled = true,
}: {
  searchQuery?: string;
  locations?: ModelTraceSearchLocation[];
  sqlWarehouseId?: string;
  enabled?: boolean;
}): { data: ModelTraceInfoV3 | undefined; isLoading: boolean } => {
  const traceIdInfo = useMemo(() => {
    if (!searchQuery) return undefined;
    return extractTraceIdFromSearchQuery(searchQuery);
  }, [searchQuery]);

  const isQueryEnabled = enabled && !isNil(traceIdInfo);
  const usingV4APIs = locations?.some(isV4TraceLocation) && shouldUseTracesV4API();

  const result = useQuery<ModelTraceInfoV3 | undefined, NetworkRequestError>({
    refetchOnWindowFocus: false,
    enabled: isQueryEnabled,
    queryKey: [TRACE_ID_LOOKUP_QUERY_KEY, traceIdInfo?.backendTraceId, traceIdInfo?.traceLocation],
    queryFn: async () => {
      if (!traceIdInfo) return undefined;

      const { backendTraceId, traceLocation: traceLocationString } = traceIdInfo;

      try {
        if (usingV4APIs && traceLocationString) {
          // Only look up traces in locations linked to the current experiment
          const isLinkedLocation = locations?.some(
            (loc) => createTraceV4SerializedLocation(loc) === traceLocationString,
          );
          if (!isLinkedLocation) return undefined;

          const traceLocation = parseTraceV4SerializedLocation(traceLocationString);
          const response = await TracesServiceV4.getBatchTracesV4({
            traceIds: [backendTraceId],
            traceLocation,
          });
          return response?.traces?.[0]?.trace_info;
        } else if (usingV4APIs && locations && locations.length > 0) {
          // For V4 APIs without a location in the trace ID, try each location
          let lastError: unknown;
          for (const location of locations) {
            try {
              const response = await TracesServiceV4.getBatchTracesV4({
                traceIds: [backendTraceId],
                traceLocation: location,
              });
              if (response?.traces?.[0]?.trace_info) {
                return response.traces[0].trace_info;
              }
            } catch (error) {
              lastError = error;
            }
          }
          // If every location returned empty, the trace wasn't found
          if (!lastError) return undefined;
          // If every location errored, throw the last one so react-query
          // marks it as failed rather than caching undefined as success
          throw lastError;
        } else {
          // For V3 APIs, use the V3 fetch
          const response = await fetchTraceInfoV3({ traceId: backendTraceId });
          return response?.trace?.trace_info;
        }
      } catch (error) {
        if (error instanceof Error && 'status' in error && (error as NetworkRequestError).status === 404) {
          return undefined;
        }
        throw error;
      }
    },
    retry: false,
  });

  return {
    data: result.data ?? undefined,
    // Only report loading when the query is actually enabled
    isLoading: isQueryEnabled && result.isLoading,
  };
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
}: {
  locations: ModelTraceSearchLocation[];
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
}) => {
  const intl = useIntl();
  const filter = createMlflowSearchFilter(runUuid, timeRange, networkFilters, filterByLoggedModelId);
  const usingV4APIs = locations?.some(isV4TraceLocation) && shouldUseTracesV4API();

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
    () => (filterByAssessmentSourceRun ? filterTracesByAssessmentSourceRunId(traces, runUuid) : traces),
    [traces, runUuid, filterByAssessmentSourceRun],
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
      // IS NULL / IS NOT NULL operators should always go to network filters
      // since they don't require a value and are handled by the backend
      if (filter.column === TracesTableColumnGroup.ASSESSMENT && isNullOperator(filter.operator)) {
        acc.networkFilters.push(filter);
        return acc;
      }

      // Assessment filters with undefined or 'Error' value must always be filtered client-side
      // because the backend cannot query for absence of an assessment or error state.
      // Note: filter.value is already converted from string 'undefined' to actual undefined by useFilters
      //
      // All numeric assessment filters are handled client-side because the backend
      // does not yet support numeric assessment comparisons.
      const isNumericValue =
        typeof filter.value === 'number' ||
        (typeof filter.value === 'string' && !isNaN(Number(filter.value)) && filter.value.trim() !== '');
      const isClientOnlyAssessmentFilter =
        filter.column === TracesTableColumnGroup.ASSESSMENT &&
        (filter.value === undefined || filter.value === ERROR_KEY || isNumericValue);

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
  enablePagination = true,
}: {
  locations: ModelTraceSearchLocation[];
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
   * When false, forces the eager-fetch path even when infinite pagination is globally enabled.
   * Used to disable pagination in run comparison mode where both runs need complete data
   * to join on inputs.
   */
  enablePagination?: boolean;
}): {
  data: ModelTraceInfoV3[] | undefined;
  isLoading: boolean;
  isFetching: boolean;
  error?: NetworkRequestError | Error;
  refetchMlflowTraces?: () => void;
  fetchNextPage?: () => void;
  hasNextPage?: boolean;
  isFetchingNextPage?: boolean;
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

  // When the search query looks like a trace ID, look it up directly via get_trace API
  // since trace_id is not a searchable field in the search traces API.
  const { data: traceIdLookupResult, isLoading: isTraceIdLookupLoading } = useTraceIdLookup({
    searchQuery,
    locations,
    sqlWarehouseId,
    enabled: !disabled,
  });

  const {
    data: traces,
    isLoading: isInnerLoading,
    isFetching: isInnerFetching,
    error,
    refetch: refetchMlflowTraces,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useSearchMlflowTracesInner({
    locations,
    filter,
    enabled: !disabled,
    pageSize,
    limit,
    orderBy,
    loggedModelId,
    sqlWarehouseId,
    enablePagination,
  });

  // Merge the trace ID lookup result with the search results. If the lookup found a trace,
  // prepend it to the list (if not already present) so it appears in the results.
  const tracesWithIdLookup = useMemo(() => {
    if (!traceIdLookupResult || !traces) {
      return traces;
    }
    const alreadyPresent = traces.some((t) => t.trace_id === traceIdLookupResult.trace_id);
    if (alreadyPresent) {
      return traces;
    }
    return [traceIdLookupResult, ...traces];
  }, [traces, traceIdLookupResult]);

  // TODO: Remove this once mlflow apis support filtering
  const evalTraceComparisonEntries = useMemo(() => {
    if (!tracesWithIdLookup) {
      return undefined;
    }

    return tracesWithIdLookup.map((trace) => {
      return {
        currentRunValue: convertTraceInfoV3ToRunEvalEntry(trace),
        otherRunValue: undefined,
      };
    });
  }, [tracesWithIdLookup]);

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
        filterOperator: filter.operator as FilterOperator,
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
    () => (filterByAssessmentSourceRun ? filterTracesByAssessmentSourceRunId(filteredTraces, runUuid) : filteredTraces),
    [filteredTraces, runUuid, filterByAssessmentSourceRun],
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
    isLoading: isInnerLoading || isTraceIdLookupLoading,
    isFetching: isInnerFetching,
    error: error || undefined,
    refetchMlflowTraces,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
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
  locations?: ModelTraceSearchLocation[];
  filter?: string;
  pageSize?: number;
  limit?: number;
  orderBy?: string[];
  loggedModelId?: string;
  sqlWarehouseId?: string;
}): Promise<ModelTraceInfoV3[]> => {
  const usingV4APIs = locations?.some(isV4TraceLocation) && shouldUseTracesV4API();

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
    const json = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/traces/search'), {
      method: 'POST',
      body: payload,
      signal,
    })) as { traces: ModelTraceInfoV3[]; next_page_token?: string };
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

interface UseSearchMlflowTracesInnerParams {
  locations?: ModelTraceSearchLocation[];
  filter?: string;
  pageSize?: number;
  limit?: number;
  orderBy?: string[];
  loggedModelId?: string;
  sqlWarehouseId?: string;
  enabled?: boolean;
  enablePagination?: boolean;
}

interface UseSearchMlflowTracesInnerResult {
  data: ModelTraceInfoV3[] | undefined;
  isLoading: boolean;
  isFetching: boolean;
  error?: NetworkRequestError | Error | null;
  refetch: () => void;
  fetchNextPage?: () => void;
  hasNextPage?: boolean;
  isFetchingNextPage?: boolean;
}

/**
 * Query cache config for trace search. Exported for tests.
 * keepPreviousData in both modes prevents the trace list from "bouncing" (disappearing
 * and showing a full loading skeleton) when search/filter changes.
 */
export function getSearchMlflowTracesQueryCacheConfig(usingV4APIs: boolean) {
  return {
    keepPreviousData: true,
    refetchOnWindowFocus: false,
    // For V4 APIs, we use server-side filtering for all filters. Since it's server-side, we
    // do not need to cache indefinitely.
    // In previous APIs, we relied on client-side filtering so we want to cache indefinitely.
    ...(usingV4APIs ? {} : { staleTime: Infinity, cacheTime: Infinity }),
  };
}

const SEARCH_TRACES_INFINITE_PAGE_SIZE = 100;

type SearchMlflowTracesResponse = {
  traces: ModelTraceInfoV3[];
  next_page_token?: string;
};

/**
 * Fetches traces using useInfiniteQuery, loading one page at a time.
 * Enabled only when shouldUseInfinitePaginatedTraces() is true.
 */
const useSearchMlflowTracesInfinite = ({
  locations,
  filter,
  orderBy,
  loggedModelId,
  sqlWarehouseId,
  enabled = true,
}: Omit<UseSearchMlflowTracesInnerParams, 'limit' | 'pageSize'>): UseSearchMlflowTracesInnerResult => {
  const { data, isLoading, isFetching, isFetchingNextPage, fetchNextPage, hasNextPage, refetch, error } =
    useInfiniteQuery<SearchMlflowTracesResponse, NetworkRequestError>({
      keepPreviousData: true,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      cacheTime: Infinity,
      enabled,
      queryKey: [
        SEARCH_MLFLOW_TRACES_QUERY_KEY,
        'infinite',
        {
          locations,
          filter,
          orderBy,
          loggedModelId,
          sqlWarehouseId,
        },
      ],
      queryFn: async ({ signal, pageParam }) => {
        const payload: SearchMlflowTracesRequest = {
          locations,
          filter,
          max_results: SEARCH_TRACES_INFINITE_PAGE_SIZE,
          order_by: orderBy,
        };
        if (loggedModelId && sqlWarehouseId) {
          payload.model_id = loggedModelId;
          payload.sql_warehouse_id = sqlWarehouseId;
        }
        if (pageParam) {
          payload.page_token = pageParam;
        }
        return fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/traces/search'), {
          method: 'POST',
          body: payload,
          signal,
        }) as Promise<SearchMlflowTracesResponse>;
      },
      getNextPageParam: (lastPage) => lastPage.next_page_token,
    });

  const allTraces = useMemo(() => data?.pages.flatMap((page) => page.traces).filter(Boolean), [data]);

  return {
    data: allTraces,
    isLoading,
    isFetching,
    error,
    refetch,
    fetchNextPage,
    hasNextPage: hasNextPage ?? false,
    isFetchingNextPage,
  };
};

/**
 * Fetches all mlflow traces for a given location/filter.
 * Uses either the regular API or long-running async API based on feature flags.
 *
 * When using long-running API:
 * - Initiates an async search operation
 * - Polls for completion
 * - Returns progress information via `longRunningProgress`
 *
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
  enabled = true,
  enablePagination = true,
}: UseSearchMlflowTracesInnerParams): UseSearchMlflowTracesInnerResult => {
  const usingV4APIs = locations?.some(isV4TraceLocation) && shouldUseTracesV4API();
  const usingLongRunningAPI = usingV4APIs && shouldUseLongRunningTracesAPI();
  const usingInfinitePagination = !usingV4APIs && shouldUseInfinitePaginatedTraces() && enablePagination;

  const queryCacheConfig = useMemo(() => getSearchMlflowTracesQueryCacheConfig(Boolean(usingV4APIs)), [usingV4APIs]);

  const sqlWarehouseQueryKey = usingV4APIs ? sqlWarehouseId : undefined;

  // Infinite paginated search (only active when feature flag is enabled)
  const infiniteResult = useSearchMlflowTracesInfinite({
    locations,
    filter,
    orderBy,
    loggedModelId,
    sqlWarehouseId,
    enabled: enabled && usingInfinitePagination,
  });

  // Standard synchronous search (only active when not using long-running API or infinite pagination)
  const syncResult = useQuery<ModelTraceInfoV3[], NetworkRequestError>({
    ...queryCacheConfig,
    enabled: enabled && !usingLongRunningAPI && !usingInfinitePagination,
    queryKey: [
      SEARCH_MLFLOW_TRACES_QUERY_KEY,
      {
        locations,
        filter,
        orderBy,
        loggedModelId,
        sqlWarehouseQueryKey,
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
  });

  if (usingInfinitePagination) {
    return infiniteResult;
  }

  return syncResult;
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
    filter.push(
      // If the query is a trace ID, use a direct indexed lookup on request_id
      // instead of trace.text ILIKE which scans the spans.content column.
      // See: https://github.com/mlflow/mlflow/discussions/21193
      /^tr-[0-9a-f]{32}$/i.test(searchQuery)
        ? `attributes.request_id = '${searchQuery.toLowerCase()}'`
        : `trace.text ILIKE '%${searchQuery}%'`,
    );
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
            if (
              networkFilter.operator === FilterOperator.IS_NULL ||
              networkFilter.operator === FilterOperator.IS_NOT_NULL
            ) {
              filter.push(`${fieldName} ${networkFilter.operator}`);
            } else {
              filter.push(`${fieldName} ${networkFilter.operator} '${networkFilter.value}'`);
            }
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
        case SESSION_COLUMN_ID:
          if (networkFilter.operator === 'CONTAINS') {
            filter.push(`request_metadata.mlflow.trace.session ILIKE '%${networkFilter.value}%'`);
          } else {
            filter.push(`request_metadata.mlflow.trace.session = '${networkFilter.value}'`);
          }
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
          // Handle IS NULL / IS NOT NULL operators for assessments
          // Note: This is not supported in managed backend
          if (
            networkFilter.operator === FilterOperator.IS_NULL ||
            networkFilter.operator === FilterOperator.IS_NOT_NULL
          ) {
            filter.push(`feedback.\`${networkFilter.key}\` ${networkFilter.operator}`);
          } else if (networkFilter.value !== 'undefined') {
            // Skip 'undefined' values - these must be filtered client-side since they represent
            // absence of an assessment, which cannot be queried on the backend
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
        case SPAN_STATUS_COLUMN_ID:
          // Span status uses exact match (OK, ERROR, UNSET)
          filter.push(`span.status ${networkFilter.operator} '${networkFilter.value}'`);
          break;
        case SPAN_CONTENT_COLUMN_ID:
          if (networkFilter.operator === 'CONTAINS') {
            filter.push(`span.content ILIKE '%${networkFilter.value}%'`);
          }
          break;
        case ISSUE_ID_COLUMN_ID:
          filter.push(`${ISSUE_ID_COLUMN_ID} ${networkFilter.operator} '${networkFilter.value}'`);
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

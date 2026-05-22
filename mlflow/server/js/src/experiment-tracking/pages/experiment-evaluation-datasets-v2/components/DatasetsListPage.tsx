import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert, DangerModal, type InputRef, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useNavigate, useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import type { Dataset } from '../hooks/useDatasetsQueries';
import { DatasetsListEmptyState, DatasetsListNoResultsEmptyState } from './DatasetsListEmptyState';
import { DatasetsListPageSkeleton } from './DatasetsListPageSkeleton';
import { DatasetsListTable } from './DatasetsListTable';
import { DatasetsListToolbar } from './DatasetsListToolbar';
import { useDatasetsPageQuery } from '../hooks/useDatasetsPageQuery';
import { useDatasetNotifications } from '../hooks/useDatasetNotifications';
import { useDatasetDelete } from '../hooks/useDatasetDelete';
import { useSlashFocusSearch } from '../hooks/useSlashFocusSearch';
import { pollUntilDone } from '../utils/pollUntilDone';
import { DEFAULT_DATASET_PAGE_SIZE } from '../utils/constants';

interface DatasetsListPageProps {
  experimentId: string;
}

const Q_PARAM = 'q';

export const DatasetsListPage = ({ experimentId }: DatasetsListPageProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const { notify, notificationContainer } = useDatasetNotifications();
  const searchInputRef = useRef<InputRef>(null);
  useSlashFocusSearch(searchInputRef);

  const [searchValue, setSearchValue] = useSearchParams((params) => params.get(Q_PARAM) ?? '');
  const setSearch = useCallback(
    (next: string) => {
      setSearchValue((params) => {
        if (next) {
          params.set(Q_PARAM, next);
        } else {
          params.delete(Q_PARAM);
        }
        return params;
      });
    },
    [setSearchValue],
  );

  // Local input value drives the controlled `<Input>`. It only gets committed to URL state
  // (and thus only triggers a fetch) on explicit submit (Enter) or clear (X). The dataset
  // search endpoint is rate-limited, so we deliberately avoid an as-you-type/debounce model.
  const [searchInputValue, setSearchInputValue] = useState(searchValue);
  // Resync the input when URL `searchValue` changes for reasons other than a direct submit —
  // browser back/forward, or a programmatic clear elsewhere on the page.
  useEffect(() => {
    setSearchInputValue(searchValue);
  }, [searchValue]);

  const handleSubmitSearch = useCallback(() => {
    setSearch(searchInputValue);
  }, [searchInputValue, setSearch]);

  const handleClearSearch = useCallback(() => {
    setSearchInputValue('');
    setSearch('');
  }, [setSearch]);

  // Cursor history — pageToken stack. `[undefined]` is page 1. Tokens stay in component state
  // because they're opaque, can expire, and would clutter the URL.
  const [tokenHistory, setTokenHistory] = useState<(string | undefined)[]>([undefined]);
  const currentPageToken = tokenHistory[tokenHistory.length - 1];

  const resetPagination = useCallback(() => setTokenHistory([undefined]), []);

  // Whenever the URL-driven `searchValue` changes — whether from a submit or from back/forward
  // navigation — drop the cursor stack. Without this, browser navigation sends a token that
  // was scoped to the previous search alongside the new (or cleared) filter.
  useEffect(() => {
    setTokenHistory([undefined]);
  }, [searchValue]);

  const { data, isLoading, isFetching, error, refetch, dataUpdatedAt } = useDatasetsPageQuery({
    experimentId,
    nameFilter: searchValue,
    pageSize: DEFAULT_DATASET_PAGE_SIZE,
    pageToken: currentPageToken,
  });

  const datasets = useMemo(() => data?.datasets ?? [], [data?.datasets]);
  const nextPageToken = data?.next_page_token;
  const hasNextPage = Boolean(nextPageToken);
  const hasPreviousPage = tokenHistory.length > 1;
  const showPagination = hasNextPage || hasPreviousPage;

  const handleNextPage = useCallback(() => {
    if (nextPageToken) {
      setTokenHistory((prev) => [...prev, nextPageToken]);
    }
  }, [nextPageToken]);

  const handlePreviousPage = useCallback(() => {
    setTokenHistory((prev) => (prev.length > 1 ? prev.slice(0, -1) : prev));
  }, []);

  // Track which (search, pageToken) tuple produced the currently-rendered data so we can
  // distinguish user-initiated query changes (search submit/clear, pagination next/prev)
  // from refresh refetches. `keepPreviousData: true` on the underlying query keeps the old
  // rows visible while a new query is in flight, which is the right UX for refresh (no
  // layout jump) but the wrong UX when the user explicitly changed the query — they expect
  // feedback that the previous result is stale. Comparing the current URL search and
  // current cursor against the last-resolved values distinguishes the two cases.
  const lastResolvedSearchRef = useRef(searchValue);
  const lastResolvedTokenRef = useRef(currentPageToken);
  useEffect(() => {
    if (!isFetching) {
      lastResolvedSearchRef.current = searchValue;
      lastResolvedTokenRef.current = currentPageToken;
    }
  }, [isFetching, searchValue, currentPageToken]);
  const isSearchInFlight = isFetching && lastResolvedSearchRef.current !== searchValue;
  const isPaginationInFlight = isFetching && lastResolvedTokenRef.current !== currentPageToken;
  const isLoadingRows = isSearchInFlight || isPaginationInFlight;

  const pollForPropagation = useCallback(
    (target: Dataset, signal: AbortSignal) =>
      pollUntilDone({
        refetch: async () => {
          const result = await refetch();
          return result.data?.datasets ?? [];
        },
        isDone: (rows) => !rows.some((d) => d.dataset_id === target.dataset_id),
        signal,
      }),
    [refetch],
  );

  const datasetDelete = useDatasetDelete({
    experimentId,
    notify,
    onMutated: resetPagination,
    pollForPropagation,
  });

  const handleDatasetCreated = useCallback(
    (dataset: Dataset) => {
      // Reset to page 1 so the newly-created dataset (which sorts to the top by created_time)
      // is visible when the user navigates back to the list.
      resetPagination();
      notify.success(
        intl.formatMessage(
          {
            defaultMessage: 'Created dataset "{name}"',
            description: 'Success toast after creating a V2 evaluation dataset',
          },
          { name: dataset.name ?? dataset.dataset_id },
        ),
      );
      navigate(Routes.getExperimentPageDatasetDetailRoute(experimentId, dataset.dataset_id));
    },
    [navigate, experimentId, notify, intl, resetPagination],
  );

  const hasActiveSearch = searchValue.trim().length > 0;
  const hasError = error instanceof Error;
  // `keepPreviousData: true` on the underlying query means `isLoading` is true only
  // before any data has resolved for any query key — the right signal for "we don't yet
  // know whether this workspace has datasets". Refetches after that point flip
  // `isFetching` instead, so the skeleton won't reappear during search or pagination.
  const isInitialLoad = isLoading;
  // CTA-only state: confirmed-empty workspace with no active search and no fetch error.
  // The CTA has its own Create-dataset button so we hide the toolbar entirely here —
  // otherwise the Create button shows twice and the disabled search bar adds noise.
  const showCtaOnly = !isInitialLoad && !hasError && datasets.length === 0 && !hasActiveSearch;
  // Inside the toolbar branch, swap the table for a no-results panel when a search
  // narrowed the list to empty. Toolbar stays visible so the user can clear the search.
  // Suppress while a user-initiated query change is in flight — the table renders skeleton
  // rows in that window, and we don't want to flash "no results" on stale data while the
  // skeletons are still settling.
  const showNoResultsPanel = !hasError && !isLoadingRows && datasets.length === 0 && hasActiveSearch;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        // Outer wrappers (PageWrapper + ExperimentPageTabs) already contribute spacing
        // on the right (24px) and bottom (8px); only top and left need padding here.
        paddingTop: theme.spacing.md,
        paddingLeft: theme.spacing.md,
        gap: theme.spacing.md,
      }}
    >
      {isInitialLoad ? (
        <DatasetsListPageSkeleton />
      ) : showCtaOnly ? (
        <DatasetsListEmptyState experimentId={experimentId} onDatasetCreated={handleDatasetCreated} refetch={refetch} />
      ) : (
        <>
          <DatasetsListToolbar
            experimentId={experimentId}
            searchInputValue={searchInputValue}
            onSearchInputChange={setSearchInputValue}
            onSearchSubmit={handleSubmitSearch}
            onSearchClear={handleClearSearch}
            onRefresh={refetch}
            isRefreshing={isFetching && !isLoading}
            lastRefreshTime={dataUpdatedAt > 0 ? dataUpdatedAt : undefined}
            onDatasetCreated={handleDatasetCreated}
            searchInputRef={searchInputRef}
          />

          {hasError && (
            <Alert
              componentId="mlflow.eval-datasets-v2.list.fetch-error"
              type="error"
              message={error.message}
              closable={false}
            />
          )}

          {showNoResultsPanel ? (
            <DatasetsListNoResultsEmptyState searchQuery={searchValue} onClearSearch={handleClearSearch} />
          ) : (
            // Scroll the table region so pagination stays reachable when the rows exceed the
            // viewport. Without this wrapper, the table grows past the outer PageWrapper's
            // `overflow: hidden` boundary and the pagination footer gets clipped.
            <div css={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
              <DatasetsListTable
                experimentId={experimentId}
                datasets={datasets}
                // Disable both pagination buttons while a pagination/search fetch is in
                // flight so the user can't queue up clicks against stale data.
                hasNextPage={hasNextPage && !isLoadingRows}
                hasPreviousPage={hasPreviousPage && !isLoadingRows}
                onNextPage={handleNextPage}
                onPreviousPage={handlePreviousPage}
                onDeleteDataset={datasetDelete.requestDelete}
                showPagination={showPagination}
                isLoadingRows={isLoadingRows}
              />
            </div>
          )}
        </>
      )}

      <DangerModal
        componentId="mlflow.eval-datasets-v2.list.delete-confirm-modal"
        visible={datasetDelete.pendingDataset !== null}
        title={
          <FormattedMessage
            defaultMessage="Delete dataset"
            description="Title for the V2 evaluation dataset delete confirmation modal"
          />
        }
        okText={intl.formatMessage({
          defaultMessage: 'Delete',
          description: 'Confirm-button text for the V2 evaluation dataset delete modal',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel-button text for the V2 evaluation dataset delete modal',
        })}
        okButtonProps={{ loading: datasetDelete.isDeleting || datasetDelete.isPolling }}
        cancelButtonProps={{ disabled: datasetDelete.isDeleting || datasetDelete.isPolling }}
        onOk={datasetDelete.confirmDelete}
        onCancel={datasetDelete.cancelDelete}
      >
        <FormattedMessage
          defaultMessage='Are you sure you want to delete the dataset "{name}"? This action cannot be undone.'
          description="Body for the V2 evaluation dataset delete confirmation modal"
          values={{
            name: datasetDelete.pendingDataset?.name ?? datasetDelete.pendingDataset?.dataset_id ?? '',
          }}
        />
        {datasetDelete.isPolling && (
          <Typography.Hint css={{ display: 'block', marginTop: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Cleaning up workspace metadata…"
              description="Status line shown in the dataset delete modal while the post-delete propagation poll is running"
            />
          </Typography.Hint>
        )}
        {datasetDelete.error && (
          <Alert
            componentId="mlflow.eval-datasets-v2.list.delete-error"
            type="error"
            message={datasetDelete.error.message}
            css={{ marginTop: theme.spacing.sm }}
            closable={false}
          />
        )}
      </DangerModal>

      {notificationContainer}
    </div>
  );
};

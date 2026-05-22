import { type RefObject } from 'react';
import { Input, type InputRef, SearchIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { CreateDatasetButton } from './CreateDatasetModal';
import type { Dataset } from '../hooks/useDatasetsQueries';
import { DatasetRefreshButton } from './DatasetRefreshButton';

export interface DatasetsListToolbarProps {
  experimentId: string;
  /**
   * Local search input value. Owned by the page; only committed to URL state on submit
   * (Enter) or clear (X) — the underlying dataset-search endpoint is rate-limited, so we
   * deliberately avoid an as-you-type/debounce model here.
   */
  searchInputValue: string;
  /** Updates the local search input value (no fetch). */
  onSearchInputChange: (next: string) => void;
  /** Commits the local input value to URL state — triggers the search fetch. */
  onSearchSubmit: () => void;
  /** Clears the local input and commits an empty value to URL state. */
  onSearchClear: () => void;
  onRefresh: () => void;
  isRefreshing: boolean;
  /** Timestamp (ms) of the last successful datasets fetch — drives the refresh tooltip. */
  lastRefreshTime: number | undefined;
  onDatasetCreated: (dataset: Dataset) => void;
  /** Forwarded to the search Input so callers can focus it (e.g. via the "/" hotkey). */
  searchInputRef?: RefObject<InputRef>;
}

export const DatasetsListToolbar = ({
  experimentId,
  searchInputValue,
  onSearchInputChange,
  onSearchSubmit,
  onSearchClear,
  onRefresh,
  isRefreshing,
  lastRefreshTime,
  onDatasetCreated,
  searchInputRef,
}: DatasetsListToolbarProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
      }}
    >
      <Input
        ref={searchInputRef}
        componentId="mlflow.eval-datasets-v2.list.search"
        prefix={<SearchIcon />}
        allowClear
        value={searchInputValue}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search datasets',
          description: 'Placeholder for the search input on the V2 evaluation datasets list page',
        })}
        aria-label={intl.formatMessage({
          defaultMessage: 'Search datasets',
          description:
            'Aria label for the search input on the V2 evaluation datasets list page (placeholder is not a label per WCAG 1.3.1)',
        })}
        onChange={(e) => onSearchInputChange(e.target.value)}
        onPressEnter={onSearchSubmit}
        onClear={onSearchClear}
        css={{ maxWidth: 360 }}
      />
      <div css={{ flex: 1 }} />
      <DatasetRefreshButton
        componentId="mlflow.eval-datasets-v2.list.refresh"
        onRefresh={onRefresh}
        isFetching={isRefreshing}
        lastRefreshTime={lastRefreshTime}
        ariaLabel={intl.formatMessage({
          defaultMessage: 'Refresh datasets',
          description: 'Aria label for the refresh button on the V2 evaluation datasets list page',
        })}
      />
      <CreateDatasetButton
        experimentId={experimentId}
        onSuccess={onDatasetCreated}
        refetch={async () => onRefresh()}
        buttonText={intl.formatMessage({
          defaultMessage: 'Create dataset',
          description: 'Primary button text for creating a new evaluation dataset on the V2 list page',
        })}
      />
    </div>
  );
};

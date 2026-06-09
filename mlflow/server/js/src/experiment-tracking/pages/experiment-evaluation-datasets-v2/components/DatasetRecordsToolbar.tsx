import { type RefObject } from 'react';
import {
  Button,
  Input,
  type InputRef,
  PlusIcon,
  SearchIcon,
  TrashIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

export interface DatasetRecordsToolbarProps {
  /** Local search input value. Owned by the controller's `useDebouncedSearchInput`. */
  searchInputValue: string;
  onSearchInputChange: (next: string) => void;
  onSearchClear: () => void;
  onRefresh: () => void;
  isRefreshing: boolean;
  /** Timestamp (ms) of the last successful records fetch — drives the refresh tooltip. */
  lastRefreshTime: number | undefined;
  onAddRecord: () => void;
  /** True while a record create is in flight — shows the Add button busy and blocks double-adds. */
  isAddingRecord?: boolean;
  /** Slot for additional toolbar controls (column selector, etc.). */
  trailingControls?: React.ReactNode;
  /** Forwarded to the search Input so callers can focus it (e.g. via the "/" hotkey). */
  searchInputRef?: RefObject<InputRef>;
  /** Number of currently bulk-selected rows. When > 0, the inline selection group appears. */
  selectionCount: number;
  /** Invoked from the inline "Delete" action when rows are selected. */
  onBulkDelete: () => void;
  /** Invoked from the inline "Clear selection" action when rows are selected. */
  onBulkClear: () => void;
}

export const DatasetRecordsToolbar = ({
  searchInputValue,
  onSearchInputChange,
  onSearchClear,
  onRefresh,
  isRefreshing,
  lastRefreshTime,
  onAddRecord,
  isAddingRecord,
  trailingControls,
  searchInputRef,
  selectionCount,
  onBulkDelete,
  onBulkClear,
}: DatasetRecordsToolbarProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const hasSelection = selectionCount > 0;

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
        componentId="mlflow.eval-datasets-v2.records.search"
        prefix={<SearchIcon />}
        allowClear
        value={searchInputValue}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search inputs, expectations…',
          description: 'Placeholder for the search input on the V2 dataset records page',
        })}
        aria-label={intl.formatMessage({
          defaultMessage: 'Search records by inputs or expectations',
          description:
            'Aria label for the search input on the V2 dataset records page (placeholder is not a label per WCAG 1.3.1)',
        })}
        onChange={(e) => onSearchInputChange(e.target.value)}
        onClear={onSearchClear}
        css={{ maxWidth: 360 }}
      />
      {trailingControls}
      {hasSelection && (
        <div
          role="region"
          aria-label={intl.formatMessage({
            defaultMessage: 'Selected records',
            description: 'Aria label for the inline bulk-selection group on the V2 dataset records toolbar',
          })}
          css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
        >
          <Button
            componentId="mlflow.eval-datasets-v2.records.selection-toolbar.delete"
            icon={<TrashIcon />}
            type="primary"
            danger
            onClick={onBulkDelete}
            // The visible "Delete (N)" label changes as the user toggles rows;
            // `aria-live` here replaces the removed "N records selected" text
            // so assistive tech still announces the count update.
            aria-live="polite"
            aria-atomic="true"
          >
            <FormattedMessage
              defaultMessage="Delete ({count})"
              description="Delete button in the V2 dataset records inline bulk-selection group"
              values={{ count: selectionCount }}
            />
          </Button>
          <Button
            componentId="mlflow.eval-datasets-v2.records.selection-toolbar.clear"
            type="tertiary"
            onClick={onBulkClear}
          >
            <FormattedMessage
              defaultMessage="Clear selection"
              description="Clear-selection button in the V2 dataset records inline bulk-selection group"
            />
          </Button>
        </div>
      )}
      <div css={{ flex: 1 }} />
      <Button
        componentId="mlflow.eval-datasets-v2.records.add-record"
        type="primary"
        icon={<PlusIcon />}
        onClick={onAddRecord}
        loading={isAddingRecord}
      >
        {intl.formatMessage({
          defaultMessage: 'Add record',
          description: 'Primary button text for adding a new dataset record on the V2 dataset records page',
        })}
      </Button>
    </div>
  );
};

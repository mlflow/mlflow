import { Alert, Button, Empty, SearchIcon, TableIcon, Typography, WarningIcon } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

// Module-local static analytics-id namespace (static componentId lint rule).
const COMPONENT_ID = 'web-shared.traces-table';

/**
 * Resolves a failed-search error to user-facing text. Injected so error copy stays product-side.
 * Load-bearing where it's optional: a consumer that omits it silently loses backend-specific hints
 * (e.g. MLflow's SQL-warehouse-timeout CTA) and falls back to the generic message below.
 */
export type ErrorDescriptionGetter = (error: unknown) => string;

/** Fallback error text used when a consumer supplies no `getErrorDescription`. */
const useDefaultErrorDescription = (): ErrorDescriptionGetter => {
  const intl = useIntl();
  return (error: unknown) => {
    const message = error instanceof Error ? error.message : undefined;
    return (
      message ||
      intl.formatMessage({
        defaultMessage: 'Something went wrong while loading traces.',
        description: 'Generic fallback description for the traces load-error surfaces',
      })
    );
  };
};

/**
 * Shared centered wrapper for the non-table states. Fixed `minHeight` keeps the region from
 * collapsing so switching between skeleton → empty → rows never jumps, and the DS `Empty` internal
 * layout is overridden to fill+center per the empty-state guidance.
 */
const CenteredState = ({ children }: { children: React.ReactNode }) => (
  <div
    css={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      minHeight: 360,
      width: '100%',
      '& > div': {
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
      },
    }}
  >
    {children}
  </div>
);

/** No traces in the current range. */
export const TracesEmptyState: React.FC = () => (
  <CenteredState>
    <Empty
      image={<TableIcon />}
      title={<FormattedMessage defaultMessage="No traces yet" description="Title for the traces empty state" />}
      description={
        <FormattedMessage
          defaultMessage="Traces logged here will appear in this table. Try widening the time range."
          description="Description for the traces empty state"
        />
      }
    />
  </CenteredState>
);

export interface TracesNoResultsStateProps {
  onClearFilters: () => void;
}

/** Filters/search matched nothing (distinct from "no traces at all"). Offers a clear affordance. */
export const TracesNoResultsState: React.FC<TracesNoResultsStateProps> = ({
  onClearFilters,
}: TracesNoResultsStateProps) => (
  <CenteredState>
    <Empty
      image={<SearchIcon />}
      title={
        <FormattedMessage
          defaultMessage="No traces match your filters"
          description="Title for the traces no-search-results state"
        />
      }
      description={
        <Typography.Link componentId={`${COMPONENT_ID}.empty.clear-filters`} onClick={onClearFilters}>
          <FormattedMessage
            defaultMessage="Clear filters"
            description="Link that clears the active search/filters on the traces table"
          />
        </Typography.Link>
      }
    />
  </CenteredState>
);

export interface TracesNoMoreResultsStateProps {
  onPrevious: () => void;
}

/**
 * Reached an empty page one step past the last full page (a cursor API can't know page N+1 is empty
 * without asking). Distinct from the initial empty state: the consumer keeps the pagination bar so
 * the user can step back, and this offers an explicit "previous page" affordance too.
 */
export const TracesNoMoreResultsState: React.FC<TracesNoMoreResultsStateProps> = ({
  onPrevious,
}: TracesNoMoreResultsStateProps) => (
  <CenteredState>
    <Empty
      image={<TableIcon />}
      title={
        <FormattedMessage defaultMessage="No more results" description="Title for the traces end-of-results state" />
      }
      description={
        <FormattedMessage
          defaultMessage="You've reached the end."
          description="Description for the traces end-of-results state"
        />
      }
      button={
        <Button componentId={`${COMPONENT_ID}.no-more-results.previous`} onClick={onPrevious}>
          <FormattedMessage
            defaultMessage="Go to previous page"
            description="Button on the traces end-of-results state that returns to the previous page"
          />
        </Button>
      }
    />
  </CenteredState>
);

export interface TracesErrorStateProps {
  onRetry: () => void;
  /** The failed search error — passed to `getErrorDescription` (if any) to derive the text. */
  error?: unknown;
  getErrorDescription?: ErrorDescriptionGetter;
}

/** First-load error (no prior rows to keep). Centered in the reserved region with a Retry action. */
export const TracesErrorState: React.FC<TracesErrorStateProps> = ({
  onRetry,
  error,
  getErrorDescription,
}: TracesErrorStateProps) => {
  const defaultDescription = useDefaultErrorDescription();
  return (
    <CenteredState>
      <Empty
        image={<WarningIcon />}
        title={
          <FormattedMessage defaultMessage="Couldn't load traces" description="Title for the traces load-error state" />
        }
        description={(getErrorDescription ?? defaultDescription)(error)}
        button={
          <Button componentId={`${COMPONENT_ID}.error.retry`} onClick={onRetry}>
            <FormattedMessage defaultMessage="Retry" description="Retry button on the traces load-error state" />
          </Button>
        }
      />
    </CenteredState>
  );
};

export interface TracesErrorAlertProps {
  /** The failed refetch error — passed to `getErrorDescription` (if any) to derive the text. */
  error?: unknown;
  onRetry: () => void;
  getErrorDescription?: ErrorDescriptionGetter;
}

/** Dismissible-free error banner shown above the table when a refetch fails but prior rows remain. */
export const TracesErrorAlert: React.FC<TracesErrorAlertProps> = ({
  error,
  onRetry,
  getErrorDescription,
}: TracesErrorAlertProps) => {
  const defaultDescription = useDefaultErrorDescription();
  return (
    <Alert
      componentId={`${COMPONENT_ID}.error-alert`}
      type="error"
      closable={false}
      message={
        <FormattedMessage
          defaultMessage="Failed to refresh traces"
          description="Title of the traces refresh-error alert"
        />
      }
      description={
        <div css={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {(getErrorDescription ?? defaultDescription)(error)}
          <Button
            componentId={`${COMPONENT_ID}.error-alert.retry`}
            size="small"
            onClick={onRetry}
            css={{ alignSelf: 'flex-start' }}
          >
            <FormattedMessage defaultMessage="Retry" description="Retry button in the traces refresh-error alert" />
          </Button>
        </div>
      }
    />
  );
};

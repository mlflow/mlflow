import { Button, RefreshIcon, SyncIcon, Tooltip } from '@databricks/design-system';
import { FormattedMessage, FormattedRelativeTime } from 'react-intl';

interface DatasetRefreshButtonProps {
  onRefresh: () => void;
  isFetching: boolean;
  /**
   * Timestamp (ms since epoch) of the last successful refresh, or `undefined` if data hasn't
   * resolved yet. Drives the "Updated N seconds ago" tooltip.
   */
  lastRefreshTime: number | undefined;
  /** Localized aria label (e.g. "Refresh datasets" or "Refresh records"). */
  ariaLabel: string;
  /** Stable analytics componentId — must differ per call site. */
  componentId: string;
}

/**
 * Refresh button that mirrors the Traces tab's pattern: shows a spinning sync icon while
 * fetching, and otherwise reveals a tooltip with a live-updating "Updated N seconds ago"
 * relative time so users know how fresh the data is without re-clicking.
 */
export const DatasetRefreshButton = ({
  onRefresh,
  isFetching,
  lastRefreshTime,
  ariaLabel,
  componentId,
}: DatasetRefreshButtonProps) => {
  const button = (
    <Button
      componentId={componentId}
      icon={isFetching ? <SyncIcon spin /> : <RefreshIcon />}
      onClick={onRefresh}
      disabled={isFetching}
      aria-label={ariaLabel}
    />
  );

  if (isFetching || lastRefreshTime === undefined) {
    return button;
  }

  return (
    <Tooltip
      componentId={`${componentId}.tooltip`}
      content={
        <FormattedMessage
          defaultMessage="Updated {time}"
          description="Tooltip for the V2 dataset refresh button showing how long ago the data was last fetched"
          values={{
            time: (
              <FormattedRelativeTime
                value={(lastRefreshTime - Date.now()) / 1000}
                numeric="always"
                updateIntervalInSeconds={10}
              />
            ),
          }}
        />
      }
    >
      {button}
    </Tooltip>
  );
};

import React, { useMemo } from 'react';
import {
  Button,
  CalendarIcon,
  ChevronDownIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  RangePicker,
  RefreshIcon,
  SyncIcon,
  Tooltip,
  Typography,
  type DateRange,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, FormattedRelativeTime, useIntl } from '@databricks/i18n';
import { useMonitoringConfig } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
import { useTracesV4TimeRange } from '../hooks/useTracesV4TimeRange';
import { getNamedDateFilters } from '../utils/dateUtils';

interface TracesV4DateSelectorProps {
  experimentId: string;
}

/**
 * Time-range selector for the V4 traces tab. Presets retain the existing compact dropdown; Custom
 * combines the design-system range picker with the same preset list. The standalone
 * `useTracesV4TimeRange` keeps URL and localStorage state isolated from V3.
 */
export const TracesV4DateSelector = React.memo(function TracesV4DateSelector({
  experimentId,
}: TracesV4DateSelectorProps) {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const monitoringConfig = useMonitoringConfig();

  const { timeLabel, startTime, endTime, setTimeRange } = useTracesV4TimeRange(experimentId);

  const namedDateFilters = useMemo(() => getNamedDateFilters(intl), [intl]);
  const selectedDateFilter = namedDateFilters.find((namedDateFilter) => namedDateFilter.key === timeLabel);

  const timeRangeFilterLabel = intl.formatMessage({
    defaultMessage: 'Time range',
    description: 'Label for the time range select dropdown',
  });
  const timeRangeButtonLabel = intl.formatMessage(
    {
      defaultMessage: 'Time range: {timeRange}',
      description: 'Accessible label for the traces time range dropdown trigger',
    },
    { timeRange: selectedDateFilter?.label ?? timeLabel },
  );

  const customPickerValue = useMemo<DateRange>(
    () => ({
      from: startTime ? new Date(startTime) : undefined,
      to: endTime ? new Date(endTime) : monitoringConfig.dateNow,
    }),
    [startTime, endTime, monitoringConfig.dateNow],
  );

  // Anchor relative ranges to "now" when the user picks a new preset — but only if the clock is
  // stale (> 1 min old). Within the 1-min window we reuse the existing cache instead of forcing a
  // refetch on tab nav / rapid filter churn.
  const refreshIfStale = () => {
    if (Date.now() - monitoringConfig.lastRefreshTime > 60_000) {
      monitoringConfig.refresh();
    }
  };

  return (
    <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
      <span css={{ display: 'flex', alignItems: 'center' }}>
        {timeLabel === 'CUSTOM' && (
          <RangePicker
            id="mlflow.traces-v4.date-selector.custom"
            includeTime
            allowClear={false}
            onChange={(event) => {
              const nextValue = event.target.value;
              if (!nextValue) {
                return;
              }
              setTimeRange({
                timeLabel: 'CUSTOM',
                startTime: nextValue.from?.toISOString(),
                endTime: nextValue.to?.toISOString(),
              });
            }}
            startDatePickerProps={{
              componentId: 'mlflow.traces-v4.date-selector.start-date-picker',
              datePickerProps: { disabled: { after: monitoringConfig.dateNow } },
              value: customPickerValue.from,
            }}
            endDatePickerProps={{
              componentId: 'mlflow.traces-v4.date-selector.end-date-picker',
              datePickerProps: { disabled: { after: monitoringConfig.dateNow } },
              value: customPickerValue.to,
            }}
          />
        )}
        <DialogCombobox
          key={String(timeLabel === 'CUSTOM')}
          id={timeLabel === 'CUSTOM' ? 'mlflow.traces-v4.date-selector.custom-presets' : undefined}
          componentId="mlflow.traces-v4.date-selector"
          label={timeRangeFilterLabel}
          value={timeLabel === 'CUSTOM' ? [] : [timeLabel]}
        >
          {timeLabel === 'CUSTOM' ? (
            <DialogComboboxCustomButtonTriggerWrapper>
              <Button
                componentId="mlflow.traces-v4.date-selector.custom-presets.trigger"
                icon={<CalendarIcon />}
                aria-label={timeRangeFilterLabel}
              />
            </DialogComboboxCustomButtonTriggerWrapper>
          ) : (
            <DialogComboboxCustomButtonTriggerWrapper>
              <Button
                componentId="mlflow.traces-v4.date-selector.trigger"
                endIcon={<ChevronDownIcon />}
                aria-label={timeRangeButtonLabel}
                data-testid="time-range-select-dropdown"
              >
                {selectedDateFilter?.triggerLabel ?? selectedDateFilter?.label ?? timeLabel}
              </Button>
            </DialogComboboxCustomButtonTriggerWrapper>
          )}
          <DialogComboboxContent width={timeLabel === 'CUSTOM' ? 330 : undefined}>
            <DialogComboboxOptionList>
              {namedDateFilters
                .filter((namedDateFilter) => timeLabel !== 'CUSTOM' || namedDateFilter.key !== 'CUSTOM')
                .map((namedDateFilter) => (
                  <DialogComboboxOptionListSelectItem
                    key={namedDateFilter.key}
                    checked={timeLabel !== 'CUSTOM' && timeLabel === namedDateFilter.key}
                    title={namedDateFilter.label}
                    value={namedDateFilter.key}
                    onChange={() => {
                      // Carry the current bounds through so switching into CUSTOM keeps the last range.
                      setTimeRange({ timeLabel: namedDateFilter.key, startTime, endTime });
                      refreshIfStale();
                    }}
                  >
                    {namedDateFilter.label}
                  </DialogComboboxOptionListSelectItem>
                ))}
            </DialogComboboxOptionList>
          </DialogComboboxContent>
        </DialogCombobox>
      </span>
    </div>
  );
});

interface TracesV4RefreshButtonProps {
  isFetching: boolean;
}

/**
 * Refresh button for the V4 traces tab (with an "X min ago" relative label). A local copy of the
 * shared v3 `TracesV3RefreshButton`, siloed in the V4 directory so the tab has no cross-imports
 * back into traces-v3.
 */
export const TracesV4RefreshButton = React.memo(function TracesV4RefreshButton({
  isFetching,
}: TracesV4RefreshButtonProps) {
  const { theme } = useDesignSystemTheme();
  const monitoringConfig = useMonitoringConfig();

  const button = (
    <Button
      type="tertiary"
      icon={isFetching ? <SyncIcon spin /> : <RefreshIcon />}
      componentId="mlflow.traces-v4.refresh-date-button"
      disabled={isFetching}
      onClick={() => {
        monitoringConfig.refresh();
      }}
      css={{
        '&, &:hover, &:focus, &:active': {
          color: `${theme.colors.textSecondary} !important`,
        },
        '& svg': {
          color: `${theme.colors.textSecondary} !important`,
        },
      }}
    >
      {!isFetching && (
        <Typography.Text color="secondary">
          <FormattedRelativeTime
            value={(monitoringConfig.lastRefreshTime - Date.now()) / 1000}
            numeric="auto"
            updateIntervalInSeconds={10}
          />
        </Typography.Text>
      )}
    </Button>
  );

  if (isFetching) {
    return button;
  }

  return (
    <Tooltip
      componentId="mlflow.traces-v4.refresh-date-button.tooltip"
      content={
        <FormattedMessage
          defaultMessage="Updated {time}"
          description="Tooltip for the refresh button showing how long ago the data was last updated"
          values={{
            time: (
              <FormattedRelativeTime
                value={(monitoringConfig.lastRefreshTime - Date.now()) / 1000}
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
});

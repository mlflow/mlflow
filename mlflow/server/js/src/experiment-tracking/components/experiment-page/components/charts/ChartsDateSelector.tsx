import React, { useMemo } from 'react';
import {
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  RefreshIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { getNamedDateFilters } from '../traces-v3/utils/dateUtils';
import {
  DEFAULT_START_TIME_LABEL,
  useMonitoringFilters,
} from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import { isNil } from 'lodash';
import { RangePicker } from '@databricks/design-system/development';
import { useMonitoringConfig } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';

export const ChartsDateSelector = React.memo(() => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const [monitoringFilters, setMonitoringFilters] = useMonitoringFilters();

  const namedDateFilters = useMemo(() => getNamedDateFilters(intl), [intl]);

  const currentStartTimeFilterLabel = intl.formatMessage({
    defaultMessage: 'Time Range',
    description: 'Label for the time range select dropdown for charts view',
  });

  const monitoringConfig = useMonitoringConfig();

  return (
    <div
      css={{
        display: 'flex',
        gap: theme.spacing.sm,
        alignItems: 'center',
      }}
    >
      <DialogCombobox
        componentId="mlflow.experiment-charts.date-selector"
        label={currentStartTimeFilterLabel}
        value={monitoringFilters.startTimeLabel ? [monitoringFilters.startTimeLabel] : [DEFAULT_START_TIME_LABEL]}
      >
        <DialogComboboxTrigger
          renderDisplayedValue={(value) => {
            return namedDateFilters.find((namedDateFilter) => namedDateFilter.key === value)?.label;
          }}
          allowClear={
            !isNil(monitoringFilters.startTimeLabel) && monitoringFilters.startTimeLabel !== DEFAULT_START_TIME_LABEL
          }
          onClear={() => {
            setMonitoringFilters({ startTimeLabel: DEFAULT_START_TIME_LABEL });
          }}
          data-testid="charts-time-range-select-dropdown"
        />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            {namedDateFilters.map((namedDateFilter) => (
              <DialogComboboxOptionListSelectItem
                key={namedDateFilter.key}
                checked={
                  monitoringFilters.startTimeLabel === namedDateFilter.key ||
                  (namedDateFilter.key === DEFAULT_START_TIME_LABEL && isNil(monitoringFilters.startTimeLabel))
                }
                title={namedDateFilter.label}
                data-testid={`charts-time-range-select-${namedDateFilter}`}
                value={namedDateFilter.key}
                onChange={() => {
                  setMonitoringFilters({
                    ...monitoringFilters,
                    startTimeLabel: namedDateFilter.key,
                  });
                }}
              >
                {namedDateFilter.label}
              </DialogComboboxOptionListSelectItem>
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>
      {monitoringFilters.startTimeLabel === 'CUSTOM' && (
        <>
          <RangePicker
            id="charts-date-picker-range"
            includeTime
            selected={{
              from: new Date(monitoringFilters.startTime!),
              to: monitoringFilters.endTime ? new Date(monitoringFilters.endTime) : monitoringConfig.dateNow,
            }}
            onChange={(e) => {
              const date = e.target.value;
              setMonitoringFilters({
                ...monitoringFilters,
                startTime: date?.from ? date.from.toISOString() : undefined,
                endTime: date?.to ? date.to.toISOString() : undefined,
              });
            }}
            startDatePickerProps={{
              componentId: 'experiment-charts-start-date-picker',
              datePickerProps: {
                disabled: {
                  after: monitoringConfig.dateNow,
                },
              },
              value: monitoringFilters.startTime ? new Date(monitoringFilters.startTime) : undefined,
            }}
            endDatePickerProps={{
              componentId: 'experiment-charts-end-date-picker',
              datePickerProps: {
                disabled: {
                  after: monitoringConfig.dateNow,
                },
              },
              value: monitoringFilters.endTime ? new Date(monitoringFilters.endTime) : undefined,
            }}
          />
        </>
      )}
      <Tooltip
        componentId="mlflow.experiment-charts.date-refresh-tooltip"
        content={intl.formatMessage(
          {
            defaultMessage: 'Showing data up to {date}.',
            description: 'Tooltip for the refresh button showing the current date and time',
          },
          {
            date: monitoringConfig.dateNow.toLocaleString(navigator.language, {
              timeZoneName: 'short',
            }),
          },
        )}
      >
        <Button
          type="link"
          componentId="mlflow.experiment-charts.refresh-date-button"
          onClick={() => {
            monitoringConfig.refresh();
          }}
        >
          <RefreshIcon />
        </Button>
      </Tooltip>
    </div>
  );
});

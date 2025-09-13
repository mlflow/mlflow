import React, { useMemo } from 'react';
import {
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  RefreshIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import {
  invalidateMlflowSearchTracesCache,
  SEARCH_MLFLOW_TRACES_QUERY_KEY,
} from '@databricks/web-shared/genai-traces-table';
import { useIntl } from '@databricks/i18n';
import { getNamedDateFilters } from './utils/dateUtils';
import {
  DEFAULT_START_TIME_LABEL,
  useMonitoringFilters,
} from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import { isNil } from 'lodash';
import { RangePicker } from '@databricks/design-system/development';
import { useMonitoringConfig } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
import { useQueryClient, useIsFetching } from '@databricks/web-shared/query-client';

export interface DateRange {
  startDate: string;
  endDate: string;
}

export const TracesV3DateSelector = React.memo(() => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const isFetching = useIsFetching({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] });

  const [monitoringFilters, setMonitoringFilters] = useMonitoringFilters();

  const namedDateFilters = useMemo(() => getNamedDateFilters(intl), [intl]);

  // List of labels for "start time" filter
  const currentStartTimeFilterLabel = intl.formatMessage({
    defaultMessage: 'Time Range',
    description: 'Label for the start range select dropdown for experiment runs view',
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
        componentId="mlflow.experiment-evaluation-monitoring.date-selector"
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
          data-testid="time-range-select-dropdown"
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
                data-testid={`time-range-select-${namedDateFilter}`}
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
            id="date-picker-range"
            includeTime
            selected={{
              // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
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
              componentId: 'experiment-evaluation-monitoring-start-date-picker',
              datePickerProps: {
                disabled: {
                  after: monitoringConfig.dateNow,
                },
              },
              value: monitoringFilters.startTime ? new Date(monitoringFilters.startTime) : undefined,
            }}
            endDatePickerProps={{
              componentId: 'experiment-evaluation-monitoring-end-date-picker',
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
        componentId="mlflow.experiment-evaluation-monitoring.trace-info-hover-request-time"
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
          componentId="mlflow.experiment-evaluation-monitoring.refresh-date-button"
          disabled={Boolean(isFetching)}
          onClick={() => {
            monitoringConfig.refresh();
            invalidateMlflowSearchTracesCache({ queryClient });
          }}
        >
          <RefreshIcon />
        </Button>
      </Tooltip>
    </div>
  );
});

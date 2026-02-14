import React, { useMemo } from 'react';
import {
  Button,
  ChevronDownIcon,
  ClockIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  RefreshIcon,
  Tooltip,
  useDesignSystemTheme,
  XCircleFillIcon,
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

type TracesV3DateSelectorRefreshButtonComponentId =
  | 'mlflow.experiment-evaluation-monitoring.refresh-date-button'
  | 'mlflow.experiment.overview.refresh-button';

interface TracesV3DateSelectorProps {
  /** Optional list of time label keys to exclude from the dropdown */
  excludeOptions?: string[];
  /** Optional custom componentId for the refresh button */
  refreshButtonComponentId?: TracesV3DateSelectorRefreshButtonComponentId;
}

const DEFAULT_REFRESH_BUTTON_COMPONENT_ID: TracesV3DateSelectorRefreshButtonComponentId =
  'mlflow.experiment-evaluation-monitoring.refresh-date-button';

// eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
export const TracesV3DateSelector = React.memo(function TracesV3DateSelector({
  excludeOptions,
  refreshButtonComponentId = DEFAULT_REFRESH_BUTTON_COMPONENT_ID,
}: TracesV3DateSelectorProps) {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const isFetching = useIsFetching({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] });

  const [monitoringFilters, setMonitoringFilters] = useMonitoringFilters();

  const namedDateFilters = useMemo(() => {
    const filters = getNamedDateFilters(intl);
    if (excludeOptions?.length) {
      return filters.filter((f) => !excludeOptions.includes(f.key));
    }
    return filters;
  }, [intl, excludeOptions]);

  // List of labels for "start time" filter
  const currentStartTimeFilterLabel = intl.formatMessage({
    defaultMessage: 'Time',
    description: 'Label for the time range select dropdown',
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
        <DialogComboboxCustomButtonTriggerWrapper>
          <Button
            componentId="mlflow.experiment-evaluation-monitoring.date-selector-button"
            icon={<ClockIcon />}
            endIcon={<ChevronDownIcon />}
            data-testid="time-range-select-dropdown"
          >
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              {namedDateFilters.find(
                (namedDateFilter) =>
                  namedDateFilter.key === monitoringFilters.startTimeLabel ||
                  (namedDateFilter.key === DEFAULT_START_TIME_LABEL && isNil(monitoringFilters.startTimeLabel)),
              )?.label}
              {!isNil(monitoringFilters.startTimeLabel) &&
                monitoringFilters.startTimeLabel !== DEFAULT_START_TIME_LABEL && (
                  <XCircleFillIcon
                    aria-hidden="false"
                    role="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                      setMonitoringFilters({ startTimeLabel: DEFAULT_START_TIME_LABEL });
                    }}
                    css={{
                      color: theme.colors.textPlaceholder,
                      fontSize: theme.typography.fontSizeSm,
                      ':hover': {
                        color: theme.colors.actionTertiaryTextHover,
                      },
                    }}
                  />
                )}
            </div>
          </Button>
        </DialogComboboxCustomButtonTriggerWrapper>
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
          componentId={refreshButtonComponentId}
          disabled={Boolean(isFetching)}
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

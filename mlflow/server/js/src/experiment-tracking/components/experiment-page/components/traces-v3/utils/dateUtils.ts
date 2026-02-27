import type { START_TIME_LABEL } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import type { IntlShape } from 'react-intl';

export type TimeBucket = 'SECOND' | 'MINUTE' | 'HOUR' | 'DAY' | 'WEEK' | 'MONTH';

export interface NamedDateFilter {
  key: START_TIME_LABEL;
  label: string;
}

export function getNamedDateFilters(intl: IntlShape): NamedDateFilter[] {
  return [
    {
      key: 'LAST_HOUR',
      label: intl.formatMessage({
        defaultMessage: 'Last hour',
        description: 'Option for the start select dropdown to filter runs from the last hour',
      }),
    },
    {
      key: 'LAST_24_HOURS',
      label: intl.formatMessage({
        defaultMessage: 'Last 24 hours',
        description: 'Option for the start select dropdown to filter runs from the last 24 hours',
      }),
    },
    {
      key: 'LAST_7_DAYS',
      label: intl.formatMessage({
        defaultMessage: 'Last 7 days',
        description: 'Option for the start select dropdown to filter runs from the last 7 days',
      }),
    },
    {
      key: 'LAST_30_DAYS',
      label: intl.formatMessage({
        defaultMessage: 'Last 30 days',
        description: 'Option for the start select dropdown to filter runs from the last 30 days',
      }),
    },
    {
      key: 'LAST_YEAR',
      label: intl.formatMessage({
        defaultMessage: 'Last year',
        description: 'Option for the start select dropdown to filter runs since the last 1 year',
      }),
    },
    {
      key: 'ALL',
      label: intl.formatMessage({
        defaultMessage: 'All',
        description: 'Option for the start select dropdown to filter runs from the beginning of time',
      }),
    },
    {
      key: 'CUSTOM',
      label: intl.formatMessage({
        defaultMessage: 'Custom',
        description: 'Option for the start select dropdown to filter runs with a custom time range',
      }),
    },
  ];
}

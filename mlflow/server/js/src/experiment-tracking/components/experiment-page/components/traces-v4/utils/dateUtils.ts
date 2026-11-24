import type { IntlShape, MessageDescriptor } from 'react-intl';

import { defineMessage } from '@databricks/i18n';

import type { TracesV4TimeLabel } from './timeRange';

type TracesV4PresetTimeLabel = Exclude<TracesV4TimeLabel, 'CUSTOM'>;

interface TracesV4DateRangePreset {
  name: TracesV4PresetTimeLabel;
  unformattedName: MessageDescriptor;
  triggerUnformattedName?: MessageDescriptor;
}

export interface NamedDateFilter {
  key: TracesV4TimeLabel;
  label: string;
  triggerLabel?: string;
}

const TRACES_V4_DATE_RANGE_PRESETS: TracesV4DateRangePreset[] = [
  {
    name: 'LAST_5_MINUTES',
    unformattedName: defineMessage({
      defaultMessage: 'Last 5 minutes',
      description: 'Option for the traces time range dropdown to filter traces from the last 5 minutes',
    }),
  },
  {
    name: 'LAST_15_MINUTES',
    unformattedName: defineMessage({
      defaultMessage: 'Last 15 minutes',
      description: 'Option for the traces time range dropdown to filter traces from the last 15 minutes',
    }),
  },
  {
    name: 'LAST_HOUR',
    unformattedName: defineMessage({
      defaultMessage: 'Last hour',
      description: 'Option for the start select dropdown to filter runs from the last hour',
    }),
  },
  {
    name: 'LAST_24_HOURS',
    unformattedName: defineMessage({
      defaultMessage: 'Last 24 hours',
      description: 'Option for the start select dropdown to filter runs from the last 24 hours',
    }),
  },
  {
    name: 'LAST_7_DAYS',
    unformattedName: defineMessage({
      defaultMessage: 'Last 7 days',
      description: 'Option for the start select dropdown to filter runs from the last 7 days',
    }),
  },
  {
    name: 'LAST_30_DAYS',
    unformattedName: defineMessage({
      defaultMessage: 'Last 30 days',
      description: 'Option for the start select dropdown to filter runs from the last 30 days',
    }),
  },
  {
    name: 'LAST_YEAR',
    unformattedName: defineMessage({
      defaultMessage: 'Last year',
      description: 'Option for the start select dropdown to filter runs since the last 1 year',
    }),
  },
  {
    name: 'ALL',
    unformattedName: defineMessage({
      defaultMessage: 'All',
      description: 'Option for the start select dropdown to filter runs from the beginning of time',
    }),
    triggerUnformattedName: defineMessage({
      defaultMessage: 'All time',
      description: 'Compact trigger label for the traces time range dropdown showing all traces',
    }),
  },
];

const CUSTOM_RANGE_MESSAGE = defineMessage({
  defaultMessage: 'Custom',
  description: 'Option for the start select dropdown to filter runs with a custom time range',
});

/** V4 preset-dropdown options, including its separate Custom option and compact trigger labels. */
export function getNamedDateFilters(intl: IntlShape): NamedDateFilter[] {
  return [
    ...TRACES_V4_DATE_RANGE_PRESETS.map(({ name, unformattedName, triggerUnformattedName }) => ({
      key: name,
      label: intl.formatMessage(unformattedName),
      triggerLabel: triggerUnformattedName ? intl.formatMessage(triggerUnformattedName) : undefined,
    })),
    {
      key: 'CUSTOM',
      label: intl.formatMessage(CUSTOM_RANGE_MESSAGE),
    },
  ];
}

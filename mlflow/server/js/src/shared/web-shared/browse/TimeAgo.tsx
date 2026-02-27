import React from 'react';

import { Tooltip } from '@databricks/design-system';
import type { IntlShape } from 'react-intl';
import { useIntl } from 'react-intl';

// Time intervals in seconds
const SECOND = 1;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
const MONTH = 30 * DAY;
const YEAR = 365 * DAY;

type Interval = {
  seconds: number;
  timeAgoMessage: (count: number) => string;
};

const getIntervals = (intl: IntlShape): Interval[] => [
  {
    seconds: YEAR,
    timeAgoMessage: (count: number) =>
      intl.formatMessage(
        {
          defaultMessage: '{count, plural, =1 {1 year} other {# years}} ago',
          description: 'Time duration in years',
        },
        { count },
      ),
  },
  {
    seconds: MONTH,
    timeAgoMessage: (count: number) =>
      intl.formatMessage(
        {
          defaultMessage: '{count, plural, =1 {1 month} other {# months}} ago',
          description: 'Time duration in months',
        },
        { count },
      ),
  },
  {
    seconds: DAY,
    timeAgoMessage: (count: number) =>
      intl.formatMessage(
        {
          defaultMessage: '{count, plural, =1 {1 day} other {# days}} ago',
          description: 'Time duration in days',
        },
        { count },
      ),
  },
  {
    seconds: HOUR,
    timeAgoMessage: (count: number) =>
      intl.formatMessage(
        {
          defaultMessage: '{count, plural, =1 {1 hour} other {# hours}} ago',
          description: 'Time duration in hours',
        },
        { count },
      ),
  },
  {
    seconds: MINUTE,
    timeAgoMessage: (count: number) =>
      intl.formatMessage(
        {
          defaultMessage: '{count, plural, =1 {1 minute} other {# minutes}} ago',
          description: 'Time duration in minutes',
        },
        { count },
      ),
  },
  {
    seconds: SECOND,
    timeAgoMessage: (count: number) =>
      intl.formatMessage(
        {
          defaultMessage: '{count, plural, =1 {1 second} other {# seconds}} ago',
          description: 'Time duration in seconds',
        },
        { count },
      ),
  },
];

export interface TimeAgoProps {
  date: Date;
  tooltipFormatOptions?: DateTooltipOptionsType;
}

type DateTooltipOptionsType = Intl.DateTimeFormatOptions;

const DateTooltipOptions: DateTooltipOptionsType = {
  timeZoneName: 'short',
  year: 'numeric',
  month: 'numeric',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
};

export const getTimeAgoStrings = ({
  date,
  intl,
  tooltipFormatOptions = DateTooltipOptions,
}: TimeAgoProps & { intl: IntlShape }): { displayText: string; tooltipTitle: string } => {
  const now = new Date();
  const seconds = Math.round((now.getTime() - date.getTime()) / 1000);

  const locale = navigator.language || 'en-US';
  let tooltipTitle = '';
  try {
    tooltipTitle = Intl.DateTimeFormat(locale, tooltipFormatOptions).format(date);
  } catch (e) {
    // ES-1357574 Do nothing; this is not a critical path, let's just not throw an error
  }

  for (const interval of getIntervals(intl)) {
    const count = Math.floor(seconds / interval.seconds);
    if (count >= 1) {
      return { displayText: interval.timeAgoMessage(count), tooltipTitle };
    }
  }

  return {
    displayText: intl.formatMessage({
      defaultMessage: 'just now',
      description: 'Indicates a time duration that just passed',
    }),
    tooltipTitle,
  };
};

export const TimeAgo: React.FC<React.PropsWithChildren<TimeAgoProps>> = ({
  date,
  tooltipFormatOptions = DateTooltipOptions,
}) => {
  const intl = useIntl();
  const { displayText, tooltipTitle } = getTimeAgoStrings({ date, intl, tooltipFormatOptions });
  return (
    <Tooltip componentId="web-shared.time-ago" content={tooltipTitle}>
      <span>{displayText}</span>
    </Tooltip>
  );
};

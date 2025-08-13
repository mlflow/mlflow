import { isNil } from 'lodash';

import type { IntlShape } from '@databricks/i18n';
import { FormattedMessage } from '@databricks/i18n';

import { getAssessmentAggregateOverallFraction } from './AggregationUtils';
import type { AssessmentAggregates, AssessmentInfo } from '../types';
const NUM_DECIMALS_PERCENTAGE_DISPLAY = 1;

/** Display a fraction (0-1) as a percentage. */
export function displayPercentage(fraction: number, numDecimalsDisplayPercentage = NUM_DECIMALS_PERCENTAGE_DISPLAY) {
  // We wrap in a Number to remove trailing zeros (4.00 => 4).
  return Number((fraction * 100).toFixed(numDecimalsDisplayPercentage)).toString();
}

export function displayFloat(value: number | undefined | null, numDecimals = 3) {
  if (isNil(value)) {
    return 'null';
  }
  const multiplier = Math.pow(10, numDecimals);
  const result = Math.round(value * multiplier) / multiplier;
  return result.toString();
}

/**
 * Computes the overall display score for an assessment, and the change in score from the other run.
 */
export function getDisplayOverallScoreAndChange(
  intl: IntlShape,
  assessmentInfo: AssessmentInfo,
  assessmentDisplayInfo: AssessmentAggregates,
): {
  displayScore: string;
  displayScoreChange: string | undefined;
  changeDirection: 'up' | 'down' | 'none';
  aggregateType: 'average' | 'percentage-true' | 'categorical';
} {
  if (assessmentInfo.dtype === 'numeric') {
    // Compute the average score for displayScore, and the change in average for displayScoreChange.
    const currentNumericValues = assessmentDisplayInfo.currentNumericValues;
    const otherNumericValues = assessmentDisplayInfo.otherNumericValues;

    let currentAverage = NaN;
    let otherAverage = NaN;
    if (currentNumericValues) {
      currentAverage = currentNumericValues.reduce((a, b) => a + b, 0) / currentNumericValues.length;
    }
    if (otherNumericValues) {
      otherAverage = otherNumericValues.reduce((a, b) => a + b, 0) / otherNumericValues.length;
    }
    const displayScore = displayFloat(currentAverage, 2);
    const scoreChange = otherNumericValues ? currentAverage - otherAverage : undefined;
    const changeDirection = scoreChange ? (scoreChange > 0 ? 'up' : 'down') : 'none';

    const displayScoreChange = scoreChange
      ? changeDirection === 'up'
        ? `+${displayFloat(Math.abs(scoreChange), 2)}`
        : changeDirection === 'down'
        ? `-${displayFloat(Math.abs(scoreChange), 2)}`
        : '+0'
      : undefined;

    return {
      displayScore,
      displayScoreChange,
      changeDirection,
      aggregateType: 'average',
    };
  } else if (assessmentInfo.dtype === 'pass-fail' || assessmentInfo.dtype === 'boolean') {
    const numDecimalsDisplayPercentage = 0;
    const scoreFraction = getAssessmentAggregateOverallFraction(assessmentInfo, assessmentDisplayInfo.currentCounts);
    const displayScore = displayPercentage(scoreFraction, numDecimalsDisplayPercentage) + '%';
    const scoreChange = assessmentDisplayInfo.otherCounts
      ? scoreFraction - getAssessmentAggregateOverallFraction(assessmentInfo, assessmentDisplayInfo.otherCounts)
      : undefined;
    const changeDirection = scoreChange ? (scoreChange > 0 ? 'up' : 'down') : 'none';
    const displayScoreChange = scoreChange
      ? (changeDirection === 'up' || changeDirection === 'none' ? '+' : '') +
        displayPercentage(scoreChange, numDecimalsDisplayPercentage) +
        '%'
      : undefined;

    return {
      displayScore,
      displayScoreChange,
      changeDirection,
      aggregateType: 'percentage-true',
    };
  } else if (assessmentInfo.dtype === 'string') {
    const numUniqueValues = assessmentInfo.uniqueValues.size;
    return {
      displayScore:
        numUniqueValues !== 1
          ? intl.formatMessage(
              {
                defaultMessage: '{numUniqueValues} values',
                description: 'Text for number of unique values for categorical assessment',
              },
              { numUniqueValues },
            )
          : intl.formatMessage({
              defaultMessage: '1 value',
              description: 'Text for number of unique values for categorical assessment',
            }),
      displayScoreChange: '',
      changeDirection: 'none',
      aggregateType: 'categorical',
    };
  } else {
    return {
      displayScore: 'N/A',
      displayScoreChange: 'N/A',
      changeDirection: 'none',
      aggregateType: 'categorical',
    };
  }
}

export function getDisplayScore(assessmentInfo: AssessmentInfo, fraction: number) {
  return displayPercentage(fraction, 0) + '%';
}

export function getDisplayScoreChange(assessmentInfo: AssessmentInfo, scoreChange: number, asPercentage = true) {
  if (assessmentInfo.dtype === 'numeric') {
    const changeDirection = scoreChange > 0 ? 'up' : 'down';
    return changeDirection === 'up' ? `+${displayFloat(scoreChange, 2)}` : `-${displayFloat(scoreChange, 2)}`;
  } else {
    const changeDirection = scoreChange >= 0 ? 'up' : 'down';
    if (asPercentage) {
      return changeDirection === 'up'
        ? `+${displayPercentage(scoreChange, 0)}%`
        : `-${displayPercentage(scoreChange * -1, 0)}%`;
    } else {
      return changeDirection === 'up' ? `+${scoreChange}` : `-${scoreChange}`;
    }
  }
}

// This is forked from mlflow: https://src.dev.databricks.com/databricks-eng/universe/-/blob/mlflow/web/js/src/common/utils/Utils.tsx?L188
export function timeSinceStr(date: any, referenceDate = new Date()) {
  // @ts-expect-error TS(2362): The left-hand side of an arithmetic operation must... Remove this comment to see the full error message
  const seconds = Math.max(0, Math.floor((referenceDate - date) / 1000));
  let interval = Math.floor(seconds / 31536000);

  if (interval >= 1) {
    return (
      <FormattedMessage
        defaultMessage="{timeSince, plural, =1 {1 year} other {# years}} ago"
        description="Text for time in years since given date for MLflow views"
        values={{ timeSince: interval }}
      />
    );
  }
  interval = Math.floor(seconds / 2592000);
  if (interval >= 1) {
    return (
      <FormattedMessage
        defaultMessage="{timeSince, plural, =1 {1 month} other {# months}} ago"
        description="Text for time in months since given date for MLflow views"
        values={{ timeSince: interval }}
      />
    );
  }
  interval = Math.floor(seconds / 86400);
  if (interval >= 1) {
    return (
      <FormattedMessage
        defaultMessage="{timeSince, plural, =1 {1 day} other {# days}} ago"
        description="Text for time in days since given date for MLflow views"
        values={{ timeSince: interval }}
      />
    );
  }
  interval = Math.floor(seconds / 3600);
  if (interval >= 1) {
    return (
      <FormattedMessage
        defaultMessage="{timeSince, plural, =1 {1 hour} other {# hours}} ago"
        description="Text for time in hours since given date for MLflow views"
        values={{ timeSince: interval }}
      />
    );
  }
  interval = Math.floor(seconds / 60);
  if (interval >= 1) {
    return (
      <FormattedMessage
        defaultMessage="{timeSince, plural, =1 {1 minute} other {# minutes}} ago"
        description="Text for time in minutes since given date for MLflow views"
        values={{ timeSince: interval }}
      />
    );
  }
  return (
    <FormattedMessage
      defaultMessage="{timeSince, plural, =1 {1 second} other {# seconds}} ago"
      description="Text for time in seconds since given date for MLflow views"
      values={{ timeSince: seconds }}
    />
  );
}

// Function to escape CSS Special characters by adding \\ before them. Needed when inserting CSS variables.
export function escapeCssSpecialCharacters(str: string) {
  // eslint-disable-next-line no-useless-escape
  return str.replace(/([!"#$%&'()*+,\.\/:;\s<=>?@[\\\]^`{|}~])/g, '\\$1');
}

// Adapted from query-insights/utils/numberUtils
export function prettySizeWithUnit(bytes: number | null | undefined, fractionDigits?: number) {
  return prettyNumberWithUnit(bytes, 1024, ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB'], fractionDigits);
}

function prettyNumberWithUnit(
  value: string | number | null | undefined,
  divisor: number,
  units: string[] = [],
  fractionDigits?: number,
): { unit: string; value: string; numericValue: number | undefined; divisor: number } {
  let val = Number(value);

  if (isNaN(val) || !isFinite(val)) {
    return {
      value: '',
      numericValue: undefined,
      unit: '',
      divisor: 1,
    };
  }

  let unit = 0;
  let greatestDivisor = 1;

  while (val >= divisor && unit < units.length - 1) {
    val /= divisor;
    greatestDivisor *= divisor;
    unit += 1;
  }

  return {
    value: formatNumber(val, fractionDigits),
    numericValue: val,
    unit: units[unit],
    divisor: greatestDivisor,
  };
}

function formatNumber(value: number, fractionDigits = 3): string {
  return Math.round(value) !== value ? value.toFixed(fractionDigits) : value.toString();
}

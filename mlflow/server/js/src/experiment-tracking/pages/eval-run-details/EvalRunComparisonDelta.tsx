import type { KeyboardEvent, ReactNode } from 'react';
import { defineMessages, useIntl, type IntlShape } from 'react-intl';
import { Tag, Tooltip, type TagColors } from '@databricks/design-system';

const FILTER_HINT_MESSAGES = defineMessages({
  regression: {
    defaultMessage: 'Click here to filter to regressions in this run.',
    description: 'Tooltip action for a regression percentage change in an evaluation run comparison',
  },
  improvement: {
    defaultMessage: 'Click here to filter to improvements in this run.',
    description: 'Tooltip action for an improvement percentage change in an evaluation run comparison',
  },
});

export const getEvalRunComparisonRelativeChangePercent = (currentValue: number, baselineValue: number) => {
  if (baselineValue === 0) {
    return currentValue === 0 ? 0 : undefined;
  }
  return ((currentValue - baselineValue) / Math.abs(baselineValue)) * 100;
};

/** A signed percent change, or a pass-rate difference in percentage points, as shown by the comparison pills. */
export const formatEvalRunComparisonPercentChange = (intl: IntlShape, percent: number): string =>
  intl.formatNumber(percent / 100, {
    style: 'percent',
    maximumFractionDigits: 2,
    signDisplay: 'exceptZero',
  });

/** Numeric scorer values and differences show two decimals, or none when they round to a whole number. */
export const formatEvalRunComparisonNumericValue = (
  intl: IntlShape,
  value: number,
  signDisplay?: 'exceptZero',
): string => {
  const fractionDigits = Number.isInteger(Math.round(value * 100) / 100) ? 0 : 2;
  return intl.formatNumber(value, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
    signDisplay,
  });
};

export type EvalRunComparisonOutcome = 'regression' | 'improvement';

export interface EvalRunComparisonTraceFilter {
  columnId: string;
  runUuid: string;
  outcome: EvalRunComparisonOutcome;
}

export interface EvalRunComparisonDeltaProps {
  currentValue: number;
  baselineValue: number;
  /** Values already expressed as percentages (for example, pass rates) change by percentage points. */
  percentagePoints?: boolean;
  absoluteDifference: ReactNode;
  componentId: string;
  color?: TagColors;
  onNavigateToTraceView?: (filter: EvalRunComparisonTraceFilter) => void;
  filter?: EvalRunComparisonTraceFilter;
  children?: ReactNode;
}

/** Displays a percent change or percentage-point change and explains the absolute difference in a tooltip. */
export const EvalRunComparisonDelta = ({
  currentValue,
  baselineValue,
  percentagePoints = false,
  absoluteDifference,
  componentId,
  color,
  onNavigateToTraceView,
  filter,
  children,
}: EvalRunComparisonDeltaProps) => {
  const intl = useIntl();
  const changePercent = percentagePoints
    ? currentValue - baselineValue
    : getEvalRunComparisonRelativeChangePercent(currentValue, baselineValue);
  const visibleChange =
    changePercent === undefined
      ? intl.formatMessage({
          defaultMessage: 'N/A',
          description: 'Relative comparison change unavailable because the baseline value is zero',
        })
      : formatEvalRunComparisonPercentChange(intl, changePercent);
  const tooltipContent =
    changePercent === undefined ? (
      <>
        {intl.formatMessage({
          defaultMessage: 'Relative change is unavailable because the baseline is zero.',
          description: 'Tooltip explanation for a comparison against a zero baseline',
        })}{' '}
        {absoluteDifference}
      </>
    ) : (
      absoluteDifference
    );
  const isClickable = changePercent !== undefined && onNavigateToTraceView !== undefined && filter !== undefined;
  const filterHint = isClickable ? intl.formatMessage(FILTER_HINT_MESSAGES[filter.outcome]) : undefined;

  return (
    <Tooltip
      componentId={componentId}
      content={
        <>
          {tooltipContent}
          {filterHint && <> {filterHint}</>}
        </>
      }
    >
      <Tag
        componentId={componentId}
        color={color}
        role={isClickable ? 'button' : 'status'}
        aria-label={visibleChange}
        onClick={
          isClickable
            ? () => {
                if (filter) onNavigateToTraceView?.(filter);
              }
            : undefined
        }
        // The OSS design-system Tag doesn't activate on Enter or Space by itself.
        onKeyDown={
          isClickable
            ? (event: KeyboardEvent<HTMLElement>) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  if (filter) onNavigateToTraceView?.(filter);
                }
              }
            : undefined
        }
        css={{ marginRight: 0 }}
      >
        {children ?? visibleChange}
      </Tag>
    </Tooltip>
  );
};

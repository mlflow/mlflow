import { isNil } from 'lodash';
import React, { useMemo } from 'react';

import { Popover, type ThemeType } from '@databricks/design-system';
import type { IntlShape } from '@databricks/i18n';

import {
  type AssessmentAggregates,
  type AssessmentFilter,
  type AssessmentInfo,
  type AssessmentValueType,
} from '../../types';
import { ERROR_KEY, getBarChartData } from '../../utils/AggregationUtils';
import { getDisplayScore, getDisplayScoreChange } from '../../utils/DisplayUtils';

const MAX_VISIBLE_ITEMS = 4;

export const CategoricalAggregateChart = React.memo(
  ({
    theme,
    intl,
    assessmentInfo,
    assessmentAggregates,
    allAssessmentFilters,
    toggleAssessmentFilter,
    currentRunDisplayName,
    compareToRunDisplayName,
  }: {
    theme: ThemeType;
    intl: IntlShape;
    assessmentInfo: AssessmentInfo;
    assessmentAggregates: AssessmentAggregates;
    allAssessmentFilters: AssessmentFilter[];
    toggleAssessmentFilter: (
      assessmentName: string,
      filterValue: AssessmentValueType,
      run: string,
      filterType?: AssessmentFilter['filterType'],
    ) => void;
    currentRunDisplayName?: string;
    compareToRunDisplayName?: string;
  }) => {
    /** Bar data */
    const barChartData = useMemo(
      () =>
        getBarChartData(
          intl,
          theme,
          assessmentInfo,
          allAssessmentFilters,
          toggleAssessmentFilter,
          assessmentAggregates,
          currentRunDisplayName,
          compareToRunDisplayName,
        ),
      [
        intl,
        theme,
        assessmentInfo,
        allAssessmentFilters,
        toggleAssessmentFilter,
        assessmentAggregates,
        currentRunDisplayName,
        compareToRunDisplayName,
      ],
    );

    // Sort barChartData: most frequent first, ties resolved by original order, error and null entries at bottom
    const sortedBarChartData = useMemo(() => {
      // Keep original order for pass-fail and boolean assessments
      if (assessmentInfo.dtype === 'pass-fail' || assessmentInfo.dtype === 'boolean') {
        return barChartData;
      }

      return barChartData
        .map((item, originalIndex) => ({ ...item, originalIndex }))
        .sort((a, b) => {
          const aIsBottomEntry = a.name === 'null' || a.name === ERROR_KEY;
          const bIsBottomEntry = b.name === 'null' || b.name === ERROR_KEY;

          // Always put error and null entries at the bottom
          if (aIsBottomEntry && !bIsBottomEntry) return 1;
          if (bIsBottomEntry && !aIsBottomEntry) return -1;

          // Sort regular entries by frequency (current.value) in descending order
          if (!aIsBottomEntry && !bIsBottomEntry) {
            const valueComparison = (b.current?.value || 0) - (a.current?.value || 0);
            if (valueComparison !== 0) return valueComparison;
          }

          // Resolve ties by original order
          return a.originalIndex - b.originalIndex;
        });
    }, [barChartData, assessmentInfo]);

    // If there's more than MAX_VISIBLE_ITEMS, show a popover with the <MAX_VISIBLE_ITEMS>th item and the rest
    const visibleItems =
      sortedBarChartData.length > MAX_VISIBLE_ITEMS
        ? sortedBarChartData.slice(0, MAX_VISIBLE_ITEMS - 1)
        : sortedBarChartData;
    const hiddenItems =
      sortedBarChartData.length > MAX_VISIBLE_ITEMS ? sortedBarChartData.slice(MAX_VISIBLE_ITEMS - 1) : [];
    const hasMoreItems = hiddenItems.length > 0;

    const hoverBarColor = theme.colors.actionDefaultBackgroundHover;
    const selectedBarColor = theme.colors.actionDefaultBackgroundPress;

    return (
      <table>
        <tbody>
          {visibleItems.map((barData) => (
            <ChartRow
              key={barData.name}
              barData={barData}
              theme={theme}
              hoverBarColor={hoverBarColor}
              selectedBarColor={selectedBarColor}
              assessmentInfo={assessmentInfo}
            />
          ))}
          {hasMoreItems && (
            <tr>
              <td colSpan={2}>
                <Popover.Root componentId="categorical-aggregate-chart-more-items">
                  <Popover.Trigger asChild>
                    <div
                      css={{
                        cursor: 'pointer',
                        padding: `${theme.spacing.xs}px`,
                        color: theme.colors.actionLinkDefault,
                        fontSize: theme.typography.fontSizeSm,
                        fontWeight: 'normal',
                        ':hover': {
                          backgroundColor: hoverBarColor,
                        },
                        borderRadius: theme.general.borderRadiusBase,
                      }}
                    >
                      {intl.formatMessage(
                        {
                          defaultMessage: '+{count} more',
                          description:
                            'Message for the popover trigger to show more items in the categorical aggregate chart.',
                        },
                        { count: hiddenItems.length },
                      )}
                    </div>
                  </Popover.Trigger>
                  <Popover.Content
                    align="start"
                    side="bottom"
                    css={{
                      maxHeight: '200px',
                      overflowY: 'auto',
                      width: '200px',
                      minWidth: '200px',
                    }}
                  >
                    <table>
                      <tbody>
                        {hiddenItems.map((barData) => (
                          <ChartRow
                            key={barData.name}
                            barData={barData}
                            theme={theme}
                            hoverBarColor={hoverBarColor}
                            selectedBarColor={selectedBarColor}
                            assessmentInfo={assessmentInfo}
                          />
                        ))}
                      </tbody>
                    </table>
                  </Popover.Content>
                </Popover.Root>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    );
  },
);

const ChartRow = ({
  barData,
  theme,
  hoverBarColor,
  selectedBarColor,
  assessmentInfo,
}: {
  barData: any;
  theme: ThemeType;
  hoverBarColor: string;
  selectedBarColor: string;
  assessmentInfo: AssessmentInfo;
}) => {
  const isError = barData.name === ERROR_KEY;
  const isNull = barData.name === 'null';

  if (isError || isNull) {
    return (
      <tr
        key={barData.name}
        css={{
          // filtering by error is not currently supported
          cursor: barData.name !== ERROR_KEY ? 'pointer' : 'not-allowed',
          ':hover': {
            backgroundColor: hoverBarColor,
          },
          color: barData.name === ERROR_KEY ? theme.colors.textValidationWarning : theme.colors.textSecondary,
          fontWeight: 'normal',
          fontSize: theme.typography.fontSizeSm,
        }}
        onClick={barData.name !== ERROR_KEY ? barData.current.toggleFilter : undefined}
      >
        <td
          css={{
            width: '100%',
            borderTopLeftRadius: theme.general.borderRadiusBase,
            borderBottomLeftRadius: theme.general.borderRadiusBase,
            backgroundColor: `${barData.current.isSelected ? selectedBarColor : ''}`,
            paddingLeft: theme.spacing.xs,
            paddingTop: theme.spacing.xs,
            paddingBottom: theme.spacing.xs,
            fontStyle: barData.name === ERROR_KEY ? 'normal' : 'italic',
          }}
        >
          <span
            css={{
              lineHeight: `${theme.typography.fontSizeSm}px`,
            }}
          >
            {barData.name}
          </span>
        </td>
        <td
          css={{
            textAlign: 'right',
            verticalAlign: 'center',
            borderTopRightRadius: theme.general.borderRadiusBase,
            borderBottomRightRadius: theme.general.borderRadiusBase,
            backgroundColor: `${barData.current.isSelected ? selectedBarColor : ''}`,
            paddingRight: theme.spacing.xs,
            paddingTop: theme.spacing.xs,
            paddingBottom: theme.spacing.xs,
            fontStyle: 'normal',
          }}
        >
          <span
            css={{
              display: 'inline-block',
              verticalAlign: 'center',
              lineHeight: `${theme.typography.fontSizeSm}px`,
            }}
          >
            {!isNil(barData.scoreChange)
              ? getDisplayScoreChange(assessmentInfo, barData.scoreChange, false)
              : barData.current.value}
          </span>
        </td>
      </tr>
    );
  }

  return (
    <tr
      key={barData.name}
      css={{
        cursor: 'pointer',
        ':hover': {
          backgroundColor: hoverBarColor,
        },
      }}
      onClick={barData.current.toggleFilter}
    >
      <td
        css={{
          width: '100%',
          borderTopLeftRadius: theme.general.borderRadiusBase,
          borderBottomLeftRadius: theme.general.borderRadiusBase,
          backgroundColor: `${barData.current.isSelected ? selectedBarColor : ''}`,
          paddingLeft: theme.spacing.xs,
          paddingTop: theme.spacing.xs,
          paddingBottom: theme.spacing.xs,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: '2px',
            width: '100%',
          }}
        >
          <span
            css={{
              fontWeight: 'normal',
              fontSize: '10px',
              lineHeight: '10px',

              color: theme.colors.textSecondary,
            }}
          >
            {barData.name}
          </span>
          <div
            css={{
              flex: 1,
              width: '100%',
              position: 'relative',
            }}
          >
            <div
              style={{
                width: barData.current.value > 0 ? `${barData.current.fraction * 100}%` : '2px',
                borderRadius: '2px',
              }}
              css={{
                position: 'relative',
                // Allow shrinking for other items with minWidth.
                flexShrink: 1,
                transition: 'width 0.3s',
                backgroundColor: barData.backgroundColor,
                height: '10px',
                display: 'flex',

                alignItems: 'center',
              }}
            />
          </div>
        </div>
      </td>
      <td
        css={{
          textAlign: 'right',
          verticalAlign: 'bottom',
          borderTopRightRadius: theme.general.borderRadiusBase,
          borderBottomRightRadius: theme.general.borderRadiusBase,
          backgroundColor: `${barData.current.isSelected ? selectedBarColor : ''}`,
          paddingRight: theme.spacing.xs,
          paddingTop: theme.spacing.xs,
          paddingBottom: theme.spacing.xs,
        }}
      >
        <span
          css={{
            color: theme.colors.textSecondary,
            fontWeight: 'normal',
            fontSize: theme.typography.fontSizeSm,
            display: 'inline-block',
            verticalAlign: 'bottom',
            lineHeight: `${theme.typography.fontSizeSm}px`,
          }}
        >
          {!isNil(barData.scoreChange)
            ? getDisplayScoreChange(assessmentInfo, barData.scoreChange)
            : getDisplayScore(assessmentInfo, barData.current.fraction)}
        </span>
      </td>
    </tr>
  );
};

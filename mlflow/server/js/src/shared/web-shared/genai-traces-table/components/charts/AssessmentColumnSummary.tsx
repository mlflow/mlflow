import { isNil } from 'lodash';
import React, { useMemo } from 'react';

import { Tag } from '@databricks/design-system';
import type { ThemeType } from '@databricks/design-system';
import type { IntlShape } from '@databricks/i18n';

import { CategoricalAggregateChart } from './CategoricalAggregateChart';
import { NumericAggregateChart } from './NumericAggregateChart';
import {
  type AssessmentAggregates,
  type AssessmentFilter,
  type AssessmentInfo,
  type AssessmentValueType,
} from '../../types';
import { getDisplayOverallScoreAndChange } from '../../utils/DisplayUtils';

export const AssessmentColumnSummary = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    theme,
    intl,
    assessmentInfo,
    assessmentAggregates,
    allAssessmentFilters,
    toggleAssessmentFilter,
    currentRunDisplayName,
    compareToRunDisplayName,
    collapsedHeader,
    categoricalChartLayout,
    displayScoreOverride,
    baselineDeltaPill,
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
    collapsedHeader?: boolean;
    /** Layout of the categorical chart; defaults to the stacked layout. */
    categoricalChartLayout?: 'stacked' | 'inline';
    /** Shown instead of the score computed from the aggregates. */
    displayScoreOverride?: string;
    /** Shown after the score, e.g. the change from a baseline run. */
    baselineDeltaPill?: React.ReactNode;
  }) => {
    const dtypeAggregateLabel = useMemo(() => {
      if (assessmentInfo.dtype === 'pass-fail') {
        return intl.formatMessage({
          defaultMessage: 'PASS',
          description: 'Header label for pass/fail assessment type for the pass rate',
        });
      } else if (assessmentInfo.dtype === 'boolean') {
        return intl.formatMessage({
          defaultMessage: 'TRUE',
          description: 'Header label for boolean assessment type for the true rate',
        });
      } else if (assessmentInfo.dtype === 'string') {
        return intl.formatMessage({
          defaultMessage: 'STRING',
          description: 'Header label for string assessment type',
        });
      } else if (assessmentInfo.dtype === 'numeric') {
        return intl.formatMessage({
          defaultMessage: 'AVG',
          description: 'Header label for numeric assessment type for the average value',
        });
      }
      return undefined;
    }, [assessmentInfo, intl]);

    /** Overall aggregate scores */
    const { displayScore, displayScoreChange, changeDirection } = useMemo(
      () => getDisplayOverallScoreAndChange(intl, assessmentInfo, assessmentAggregates),
      [intl, assessmentInfo, assessmentAggregates],
    );

    if (assessmentInfo.dtype === 'unknown') {
      return null;
    }

    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            paddingTop: theme.spacing.xs,
          }}
        >
          {/* Aggregate label, e.g. "PASS" or "TRUE" */}
          <div
            css={{
              fontWeight: 400,
              fontSize: '10px',
              color: theme.colors.textPlaceholder,
            }}
          >
            {dtypeAggregateLabel}
          </div>
          {/* Overall score & diff */}
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.xs,
            }}
          >
            {/* Current run score */}
            <div
              css={{
                fontSize: theme.typography.fontSizeLg,
                color: theme.colors.textPrimary,
                fontWeight: theme.typography.typographyBoldFontWeight,
              }}
            >
              {displayScoreOverride ?? displayScore}
            </div>
            {baselineDeltaPill}
            {/* Diff score */}
            {!isNil(displayScoreChange) && compareToRunDisplayName && (
              <Tag
                componentId="mlflow.genai-traces-table.assessment-column-summary.score-change"
                color={changeDirection === 'up' ? 'lime' : changeDirection === 'down' ? 'coral' : undefined}
                css={{ alignSelf: 'center', marginRight: 0 }}
              >
                {displayScoreChange}
              </Tag>
            )}
          </div>
        </div>

        {!collapsedHeader &&
          (!isNil(assessmentAggregates.currentNumericAggregate) ? (
            <NumericAggregateChart numericAggregate={assessmentAggregates.currentNumericAggregate} />
          ) : (
            // Categorical charts
            <CategoricalAggregateChart
              theme={theme}
              intl={intl}
              assessmentInfo={assessmentInfo}
              assessmentAggregates={assessmentAggregates}
              allAssessmentFilters={allAssessmentFilters}
              toggleAssessmentFilter={toggleAssessmentFilter}
              currentRunDisplayName={currentRunDisplayName}
              compareToRunDisplayName={compareToRunDisplayName}
              layout={categoricalChartLayout}
            />
          ))}
      </div>
    );
  },
);

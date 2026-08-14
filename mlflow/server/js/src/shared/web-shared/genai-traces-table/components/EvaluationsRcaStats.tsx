import { useMemo } from 'react';

import { Tooltip, useDesignSystemTheme, XCircleFillIcon } from '@databricks/design-system';
import { useIntl, type IntlShape } from '@databricks/i18n';

import {
  KnownEvaluationResultAssessmentStringValue,
  KnownEvaluationResultAssessmentValueMapping,
  withAlpha,
} from './GenAiEvaluationTracesReview.utils';
import { RunColorCircle } from './RunColorCircle';
import type { AssessmentAggregates, AssessmentFilter, AssessmentInfo, AssessmentValueType } from '../types';
import { COMPARE_TO_RUN_COLOR, CURRENT_RUN_COLOR, getEvaluationResultAssessmentBackgroundColor } from '../utils/Colors';
import { displayPercentage } from '../utils/DisplayUtils';

interface RcaAssessmentRunDisplayInfoCount {
  count: number;
  fraction: number;
  percentage: string;
  tooltip: string;
  toggleFilter: () => void;
  isSelected: boolean;
}

interface RcaAssessmentDisplayInfoCount {
  assessment: string;
  title: string;
  currentInfo: RcaAssessmentRunDisplayInfoCount;
  otherInfo?: RcaAssessmentRunDisplayInfoCount;
  icon: JSX.Element;
}

export const EvaluationsRcaStats = ({
  overallAssessmentInfo,
  assessmentNameToAggregates,
  allAssessmentFilters,
  toggleAssessmentFilter,
  runUuid,
  compareToRunUuid,
}: {
  overallAssessmentInfo: AssessmentInfo;
  assessmentNameToAggregates: Record<string, AssessmentAggregates>;
  allAssessmentFilters: AssessmentFilter[];
  toggleAssessmentFilter: (
    assessmentName: string,
    filterValue: AssessmentValueType,
    run: string,
    filterType?: AssessmentFilter['filterType'],
  ) => void;
  runUuid?: string;
  compareToRunUuid?: string;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const rcaData = useMemo(() => {
    return getRcaData(
      intl,
      assessmentNameToAggregates,
      allAssessmentFilters,
      toggleAssessmentFilter,
      Boolean(compareToRunUuid),
      runUuid,
      compareToRunUuid,
    );
  }, [intl, assessmentNameToAggregates, allAssessmentFilters, toggleAssessmentFilter, compareToRunUuid, runUuid]);

  if (rcaData.length === 0) {
    return <></>;
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
          fontWeight: theme.typography.typographyBoldFontWeight,
        }}
      >
        {intl.formatMessage({
          defaultMessage: 'Root Cause Analysis',
          description: 'Root cause analysis section title',
        })}
      </div>

      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        <RcaPills
          overallAssessmentInfo={overallAssessmentInfo}
          rcaData={rcaData}
          run="current"
          runColor={compareToRunUuid ? CURRENT_RUN_COLOR : undefined}
        />
      </div>
      {compareToRunUuid && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
          }}
        >
          <RcaPills
            overallAssessmentInfo={overallAssessmentInfo}
            rcaData={rcaData}
            run="other"
            runColor={COMPARE_TO_RUN_COLOR}
          />
        </div>
      )}
    </div>
  );
};

export const RcaPills = ({
  rcaData,
  run,
  runColor,
  overallAssessmentInfo,
}: {
  rcaData: RcaAssessmentDisplayInfoCount[];
  run: 'current' | 'other';
  runColor?: string;
  overallAssessmentInfo: AssessmentInfo;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.sm, alignItems: 'center' }}>
      {runColor && (
        <div css={{ display: 'flex' }}>
          <RunColorCircle color={runColor} />
        </div>
      )}
      <div
        css={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: theme.spacing.xs,
        }}
      >
        {rcaData.map((item, i) => {
          const runDisplayInfoCount = run === 'current' ? item.currentInfo : item.otherInfo;
          if (!runDisplayInfoCount) {
            // eslint-disable-next-line react/jsx-key -- TODO(FEINF-1756)
            return <></>;
          }
          const backgroundColor = getEvaluationResultAssessmentBackgroundColor(theme, overallAssessmentInfo, {
            stringValue: 'no',
          });
          return (
            <Tooltip key={i} content={runDisplayInfoCount.tooltip} componentId="mlflow.evaluations_review.rca_pill">
              <>
                <div
                  key={i}
                  style={{
                    border: runDisplayInfoCount.isSelected ? `1px solid ${theme.colors.grey400}` : '',
                  }}
                  css={{
                    color: theme.colors.textSecondary,
                    borderRadius: theme.general.borderRadiusBase,
                    padding: `0 ${theme.spacing.xs}px`,
                    height: '20px',
                    display: 'flex',
                    backgroundColor: runDisplayInfoCount.isSelected
                      ? withAlpha(backgroundColor, 1.0)
                      : withAlpha(backgroundColor, 0.5),
                    // Bring back when we support filtering on RCA.
                    // cursor: 'pointer',
                    // '&:hover': {
                    //   backgroundColor: withAlpha(backgroundColor, 0.9),
                    // },
                    gap: theme.spacing.xs,
                    fontSize: theme.typography.fontSizeSm,
                  }}
                  // Bring back when we support filtering on RCA.
                  // onClick={runDisplayInfoCount.toggleFilter}
                >
                  {runDisplayInfoCount.count}
                  <div>{item.title.toLocaleLowerCase()}</div>
                </div>
              </>
            </Tooltip>
          );
        })}
      </div>
    </div>
  );
};

function getRcaData(
  intl: IntlShape,
  assessmentNameToAggregates: Record<string, AssessmentAggregates>,
  assessmentFilters: AssessmentFilter[],
  toggleAssessmentFilter: (
    assessmentName: string,
    filterValue: AssessmentValueType,
    run: string,
    filterType?: AssessmentFilter['filterType'],
  ) => void,
  isCompareToRun: boolean,
  currentRunDisplayName?: string,
  otherRunDisplayName?: string,
): RcaAssessmentDisplayInfoCount[] {
  // If all of the unfiltered assessment display infos have 0 root cause, return an empty array.
  if (
    Object.values(assessmentNameToAggregates).every(
      (assessmentDisplayInfo) =>
        assessmentDisplayInfo.currentNumRootCause === 0 && (assessmentDisplayInfo.otherNumRootCause || 0) === 0,
    )
  ) {
    return [];
  }
  // Remove any root cause values that are 0 in the unfiltered set. This that the set of rca pills don't hide 0 values when filtered
  // but keep the set minimal when unfiltered.
  const sortedAssessmentAggregates = Object.values(assessmentNameToAggregates)
    .filter((x) => x.currentNumRootCause > 0 || (x.otherNumRootCause || 0) > 0)
    .sort((a, b) => b.currentNumRootCause - a.currentNumRootCause);
  const maxNumRootCause = Math.max(
    sortedAssessmentAggregates[0]?.currentNumRootCause || 0,
    sortedAssessmentAggregates[0]?.otherNumRootCause || 0,
  );
  return sortedAssessmentAggregates.map((assessmentDisplayInfo) => {
    const knownValueLabel =
      KnownEvaluationResultAssessmentValueMapping[assessmentDisplayInfo.assessmentInfo.name]?.[
        KnownEvaluationResultAssessmentStringValue.NO
      ];
    const title = knownValueLabel ? intl.formatMessage(knownValueLabel) : assessmentDisplayInfo.assessmentInfo.name;

    const currentCounts = assessmentDisplayInfo.currentCounts;
    const numPassing = currentCounts?.get(KnownEvaluationResultAssessmentStringValue.YES) || 0;
    const numFailing = currentCounts?.get(KnownEvaluationResultAssessmentStringValue.NO) || 0;
    const numMissing = currentCounts?.get(undefined) || 0;
    const numRootCause = assessmentDisplayInfo.currentNumRootCause;

    const numEvals = numPassing + numFailing + numMissing;

    const otherCounts = assessmentDisplayInfo.otherCounts;
    let otherNumEvals: number | undefined;
    if (assessmentDisplayInfo.otherCounts !== undefined) {
      const otherNumPassing = otherCounts?.get(KnownEvaluationResultAssessmentStringValue.YES) || 0;
      const otherNumFailing = otherCounts?.get(KnownEvaluationResultAssessmentStringValue.NO) || 0;
      const otherNumMissing = otherCounts?.get(undefined) || 0;
      otherNumEvals = otherNumPassing + otherNumFailing + otherNumMissing;
    }
    const otherNumRootCause = assessmentDisplayInfo.otherNumRootCause || 0;
    const rootCauseFraction = numRootCause / numEvals;
    const otherRootCauseFraction = otherNumRootCause / (otherNumEvals || 1);
    const currentPercentage = displayPercentage(rootCauseFraction);
    const otherPercentage = displayPercentage(otherRootCauseFraction);

    return {
      assessment: assessmentDisplayInfo.assessmentInfo.name,
      // Map assessment to known values.
      title,
      currentInfo: {
        count: numRootCause,
        fraction: numRootCause / maxNumRootCause,
        tooltip: intl.formatMessage(
          {
            defaultMessage:
              '{numRootCause}/{numEvals} ({percentage}%) runs failed due to the {assessment} judge failing for the current run "{currentRun}".',
            description: 'Tooltip for the root cause metrics bar on the LLM evaluation page.',
          },
          {
            numRootCause,
            numEvals,
            percentage: currentPercentage,
            assessment: assessmentDisplayInfo.assessmentInfo.name,
            currentRun: currentRunDisplayName,
          },
        ),
        toggleFilter: () =>
          toggleAssessmentFilter(assessmentDisplayInfo.assessmentInfo.name, 'failing_root_cause', 'current'),
        isSelected: assessmentFilters.some(
          (filter) =>
            filter.filterType === 'rca' &&
            filter.run === 'current' &&
            filter.assessmentName === assessmentDisplayInfo.assessmentInfo.name,
        ),
        percentage: currentPercentage,
      },
      otherInfo: isCompareToRun
        ? {
            count: otherNumRootCause,
            fraction: otherNumRootCause / maxNumRootCause,
            tooltip: intl.formatMessage(
              {
                defaultMessage:
                  '{numRootCause}/{numEvals} ({percentage}%) runs failed due to the {assessment} judge failing for run "{otherRunDisplayName}".',
                description: 'Tooltip for the root cause metrics bar on the LLM evaluation page.',
              },
              {
                numRootCause: otherNumRootCause,
                numEvals: otherNumEvals,
                percentage: otherPercentage,
                assessment: assessmentDisplayInfo.assessmentInfo.name,
                otherRunDisplayName,
              },
            ),
            toggleFilter: () => toggleAssessmentFilter(assessmentDisplayInfo.assessmentInfo.name, 'rca', 'other'),
            isSelected: assessmentFilters.some(
              (filter) =>
                filter.filterType === 'rca' &&
                filter.run === 'other' &&
                filter.assessmentName === assessmentDisplayInfo.assessmentInfo.name,
            ),
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
            percentage: otherPercentage!,
          }
        : undefined,
      icon: <XCircleFillIcon />,
    };
  });
}

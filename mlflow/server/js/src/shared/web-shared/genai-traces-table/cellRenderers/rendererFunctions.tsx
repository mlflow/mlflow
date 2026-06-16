import type { CellContext } from '@tanstack/react-table';
import { first, isNil } from 'lodash';
import React, { useContext, useMemo } from 'react';
import type { FormatDateOptions } from 'react-intl';

import type { ThemeType } from '@databricks/design-system';
import {
  ArrowRightIcon,
  CheckCircleIcon,
  HoverCard,
  Overflow,
  Spinner,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  UserIcon,
  XCircleIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl, type IntlShape } from '@databricks/i18n';
import type { ModelTraceInfoV3 } from '../../model-trace-explorer/ModelTrace.types';
import { ExpectationValuePreview } from '../../model-trace-explorer/assessments-pane/ExpectationValuePreview';
import { useModelTraceExplorerRunJudgesContext } from '../../model-trace-explorer/contexts/RunJudgesContext';

import { GenAITracesTableContext } from '../GenAITracesTableContext';

import { ExecutionDurationTag } from './ExecutionDurationTag';
import { IssuesCell } from './IssuesCell';
import { LoggedModelCell } from './LoggedModelCell';
import { NullCell } from './NullCell';
import { RunName } from './RunName';
import { Link, generatePath } from '../utils/RoutingUtils';
import { SessionIdLinkWrapper } from './SessionIdLinkWrapper';
import { SourceCellRenderer } from './Source/SourceRenderer';
import { StackedComponents } from './StackedComponents';
import { StatusCellRenderer } from './StatusRenderer';
import { TagsCellRenderer } from './Tags/TagsCellRenderer';
import { TokensCell } from './TokensCell';
import { getTraceInfoValueWithColId } from '../GenAiTracesTable.utils';
import { compareAssessmentValues, formatResponseTitle } from '../GenAiTracesTableBody.utils';
import { readTraceTag, RESULT_ASSESSMENT_NAME } from '../utils/TraceUtils';
import { EvaluationsReviewAssessmentTag } from '../components/EvaluationsReviewAssessmentTag';
import {
  getEvaluationResultAssessmentValue,
  getEvaluationResultInputTitle,
  KnownEvaluationResultAssessmentStringValue,
  KnownEvaluationResultAssessmentValueMissingTooltip,
  stringifyValue,
} from '../components/GenAiEvaluationTracesReview.utils';
import { RunColorCircle } from '../components/RunColorCircle';
import {
  CUSTOM_METADATA_COLUMN_ID,
  EXECUTION_DURATION_COLUMN_ID,
  ISSUES_COLUMN_ID,
  LINKED_PROMPTS_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  REQUEST_TIME_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  SESSION_COLUMN_ID,
  SIMULATION_GOAL_COLUMN_ID,
  SIMULATION_PERSONA_COLUMN_ID,
  SOURCE_COLUMN_ID,
  STATE_COLUMN_ID,
  TAGS_COLUMN_ID,
  TOKENS_COLUMN_ID,
  TRACE_ID_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  USER_COLUMN_ID,
} from '../hooks/useTableColumns';
import type { AssessmentInfo, EvalTraceComparisonEntry } from '../types';
import { getUniqueValueCountsBySourceId } from '../utils/AggregationUtils';
import { COMPARE_TO_RUN_COLOR, CURRENT_RUN_COLOR } from '../utils/Colors';
import { highlightSearchInText, normalizeDurationString, timeSinceStr } from '../utils/DisplayUtils';
import { shouldEnableTagGrouping } from '../utils/FeatureUtils';
import {
  getCustomMetadataKeyFromColumnId,
  getExperimentIdFromTraceLocation,
  getTagKeyFromColumnId,
  getTraceInfoOutputs,
  MLFLOW_SOURCE_RUN_KEY,
} from '../utils/TraceUtils';

type timestampType = number | string | Date | null;

/**
 * Formats a timestamp into a date and time string.
 * @param timestamp
 * @param intl
 * @param options
 * @returns {string} formatted date and time string
 * @example
 * formatDateTime(1626825600000, intl);
 * // => 'Jul 21, 2021, 12:00 AM'
 * formatDateTime(1626825600000, intl, { hour: 'numeric', minute: '2-digit' });
 * // => 'Jul 21, 2021, 5:30 AM'
 * formatDateTime(1626825600000, intl, { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
 * // => 'Jul 21, 2021, 05:30 AM PDT'
 * formatDateTime(1626825600000, intl, { month: 'long', minute: '2-digit'});
 * // => 'July 21, 2021, 05:30 AM'
 **/
export function formatDateTime(timestamp: timestampType, intl: IntlShape, options?: FormatDateOptions): string {
  if (!timestamp) {
    return '';
  }

  return intl.formatDate(timestamp, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    ...options,
  });
}

export const assessmentCellRenderer = (
  theme: ThemeType,
  intl: IntlShape,
  isComparing: boolean,
  assessmentInfo: AssessmentInfo,
  comparisonEntry: EvalTraceComparisonEntry,
) => {
  const assessmentName = assessmentInfo.name;

  // Regression-test "Result" column: show N/M assertions passed per row, with a
  // hover-card breakdown. Only reached when the data layer synthesized "Result".
  if (assessmentName === RESULT_ASSESSMENT_NAME) {
    const breakdown = (runValue: typeof comparisonEntry.currentRunValue) => {
      const byName = runValue?.responseAssessmentsByName ?? {};
      const rows: { label: string; passed: boolean }[] = [];
      for (const name of Object.keys(byName)) {
        if (name === RESULT_ASSESSMENT_NAME) continue;
        for (const r of byName[name] ?? []) {
          const value = getEvaluationResultAssessmentValue(r);
          rows.push({
            label: name,
            passed: value === KnownEvaluationResultAssessmentStringValue.YES || value === true,
          });
        }
      }
      return rows;
    };
    const badge = (runValue: typeof comparisonEntry.currentRunValue) => {
      const rows = breakdown(runValue);
      const total = rows.length;
      if (total === 0) return null;
      const passed = rows.filter((r) => r.passed).length;
      const allPassed = passed === total;
      const tag = (
        <Tag
          componentId="mlflow.genai-traces-table.result"
          color={allPassed ? 'turquoise' : 'coral'}
          css={{
            margin: 0,
            display: 'inline-flex',
            alignItems: 'center',
            gap: 4,
            width: 'fit-content',
            cursor: 'default',
          }}
        >
          {allPassed ? (
            <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />
          ) : (
            <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />
          )}
          {passed}/{total}
        </Tag>
      );
      // Hover shows the per-assertion breakdown (name + pass/fail, no rationale).
      return (
        <HoverCard
          side="bottom"
          content={
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, maxWidth: '22rem' }}>
              {rows.map((row, i) => (
                <div key={i} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  {row.passed ? (
                    <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />
                  ) : (
                    <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />
                  )}
                  <Typography.Text css={{ wordBreak: 'break-word' }}>{row.label}</Typography.Text>
                </div>
              ))}
            </div>
          }
          trigger={tag}
        />
      );
    };
    return (
      <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
        {badge(comparisonEntry.currentRunValue)}
        {isComparing && badge(comparisonEntry.otherRunValue)}
      </div>
    );
  }

  const assessment = {
    currentValue: first(comparisonEntry.currentRunValue?.responseAssessmentsByName[assessmentName]),
    otherValue: first(comparisonEntry.otherRunValue?.responseAssessmentsByName[assessmentName]),
  };

  const uniqueValueCounts = getUniqueValueCountsBySourceId(
    assessmentInfo,
    comparisonEntry.currentRunValue?.responseAssessmentsByName[assessmentName] || [],
  );

  const currentIsAssessmentRootCause =
    comparisonEntry.currentRunValue?.overallAssessments[0]?.rootCauseAssessment?.assessmentName === assessmentName;
  const otherIsAssessmentRootCause =
    comparisonEntry.otherRunValue?.overallAssessments[0]?.rootCauseAssessment?.assessmentName === assessmentName;

  const currentValue = assessment.currentValue
    ? getEvaluationResultAssessmentValue(assessment.currentValue)
    : undefined;
  const otherValue = assessment.otherValue ? getEvaluationResultAssessmentValue(assessment.otherValue) : undefined;
  const assessmentComparison = compareAssessmentValues(
    assessmentInfo,
    isNil(currentValue) ? undefined : currentValue,
    isNil(otherValue) ? undefined : otherValue,
  );

  const assessmentChanged = otherValue !== currentValue;

  const missingTooltip = KnownEvaluationResultAssessmentValueMissingTooltip[assessmentName];
  return (
    <div
      css={{
        display: 'flex',
        gap: theme.spacing.sm,
        alignItems: 'center',
        marginTop: 'auto',
        marginBottom: 'auto',
      }}
    >
      {isComparing ? (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            borderRadius: theme.legacyBorders.borderRadiusMd,
            marginTop: 'auto',
            marginBottom: 'auto',
          }}
        >
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.sm,
              alignItems: 'center',
            }}
          >
            <EvaluationsReviewAssessmentTag
              key={`tag_${comparisonEntry.currentRunValue?.evaluationId}_${assessment.currentValue?.name}`}
              showRationaleInTooltip
              disableJudgeTypeIcon
              hideAssessmentName
              assessment={assessment.currentValue}
              isRootCauseAssessment={currentIsAssessmentRootCause}
              assessmentInfo={assessmentInfo}
              type="value"
            />
            {assessmentChanged && (
              <ArrowRightIcon
                css={{
                  // Rotate by 45 degrees when the current is passing
                  transform: assessmentComparison === 'greater' ? 'rotate(-45deg)' : 'rotate(45deg)',
                  color: theme.colors.textSecondary,
                }}
              />
            )}
          </div>
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.sm,
              alignItems: 'center',
            }}
          >
            <EvaluationsReviewAssessmentTag
              key={`tag_${comparisonEntry.otherRunValue?.evaluationId}_${assessment.currentValue?.name}`}
              showRationaleInTooltip
              disableJudgeTypeIcon
              hideAssessmentName
              assessment={assessment.otherValue}
              isRootCauseAssessment={otherIsAssessmentRootCause}
              assessmentInfo={assessmentInfo}
              type="value"
            />
            {/* This invisible icon aligns the values. */}
            {assessmentChanged && (
              <ArrowRightIcon
                css={{
                  visibility: 'hidden',
                }}
              />
            )}
          </div>
        </div>
      ) : assessment.currentValue ? (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          {uniqueValueCounts.map((uniqueValueCount) => {
            const assessment = uniqueValueCount.latestAssessment;
            const count = uniqueValueCount.count;
            return (
              <EvaluationsReviewAssessmentTag
                key={`tag_${uniqueValueCount.latestAssessment.name}_${uniqueValueCount.value}`}
                showRationaleInTooltip
                disableJudgeTypeIcon
                hideAssessmentName
                assessment={assessment}
                isRootCauseAssessment={currentIsAssessmentRootCause}
                assessmentInfo={assessmentInfo}
                type="value"
                count={count}
              />
            );
          })}
        </div>
      ) : (
        <EvaluationsReviewAssessmentTag
          key={`tag_${assessmentName}_${comparisonEntry.currentRunValue?.evaluationId}`}
          showRationaleInTooltip
          disableJudgeTypeIcon
          hideAssessmentName
          isRootCauseAssessment={currentIsAssessmentRootCause}
          assessmentInfo={assessmentInfo}
          assessment={{
            name: assessmentName,
            rationale: missingTooltip
              ? intl.formatMessage(missingTooltip)
              : intl.formatMessage({
                  defaultMessage: 'No assessment for this evaluation',
                  description: 'Text displayed when there is no assessment for a given evaluation',
                }),
            source: {
              sourceId: '',
              sourceType: 'AI_JUDGE',
              metadata: {},
            },
            stringValue: null,
            booleanValue: null,
            rootCauseAssessment: null,
            numericValue: null,
            timestamp: null,
            metadata: {},
          }}
          type="value"
        />
      )}
    </div>
  );
};

/**
 * Wrapper component for assessment cells that checks the context for session grouping.
 * Hides session-level assessments in regular rows when grouped by session.
 * Shows a spinner when a judge is currently running on this trace.
 */
export const AssessmentCell: React.FC<{
  isComparing: boolean;
  assessmentInfo: AssessmentInfo;
  comparisonEntry: EvalTraceComparisonEntry;
}> = ({ isComparing, assessmentInfo, comparisonEntry }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { isGroupedBySession } = useContext(GenAITracesTableContext);
  const { evaluations } = useModelTraceExplorerRunJudgesContext();

  const traceId = comparisonEntry.currentRunValue?.traceInfo?.trace_id;

  const isJudgeRunning = useMemo(() => {
    if (!traceId || !evaluations) return false;
    return Object.values(evaluations).some(
      (evaluation) =>
        evaluation.isLoading &&
        evaluation.label === assessmentInfo.name &&
        (!evaluation.tracesData || traceId in evaluation.tracesData),
    );
  }, [traceId, evaluations, assessmentInfo.name]);

  // Hide session-level assessments in regular rows when grouped by session
  if (isGroupedBySession && assessmentInfo.isSessionLevelAssessment) {
    return <NullCell />;
  }

  if (isJudgeRunning) {
    return (
      <Tooltip
        componentId="mlflow.genai-traces-table.assessment-cell-judge-running"
        content={intl.formatMessage({
          defaultMessage: 'Judge is running…',
          description: 'Tooltip shown in assessment cell while a judge is executing on this trace',
        })}
      >
        <span css={{ display: 'inline-flex', alignItems: 'center', color: theme.colors.textSecondary }}>
          <Spinner size="small" />
        </span>
      </Tooltip>
    );
  }

  return assessmentCellRenderer(theme, intl, isComparing, assessmentInfo, comparisonEntry);
};

export const expectationCellRenderer = (
  theme: ThemeType,
  intl: IntlShape,
  isComparing: boolean,
  expectationName: string,
  comparisonEntry: EvalTraceComparisonEntry,
) => {
  const currentValue = comparisonEntry.currentRunValue?.targets?.[expectationName];
  const otherValue = comparisonEntry.otherRunValue?.targets?.[expectationName];

  const currentValuePreview = currentValue ? (
    <ExpectationValuePreview parsedValue={currentValue} singleLine />
  ) : (
    <NullCell isComparing={isComparing} />
  );
  const otherValuePreview = otherValue ? (
    <ExpectationValuePreview parsedValue={otherValue} singleLine />
  ) : (
    <NullCell isComparing={isComparing} />
  );

  return isComparing ? (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div css={{ display: 'flex', flex: 1 }}>{currentValuePreview}</div>
      <div css={{ display: 'flex', flex: 1 }}>{otherValuePreview}</div>
    </div>
  ) : (
    currentValuePreview
  );
};

export const inputColumnCellRenderer = (
  onChangeEvaluationId: (evaluationId: string | undefined, traceInfo?: ModelTraceInfoV3) => void,
  row: CellContext<EvalTraceComparisonEntry, unknown>,
  isComparing: boolean,
  theme: ThemeType,
  inputColumn: string,
  getRunColor?: (runUuid: string) => string,
) => {
  const value = row.getValue() as EvalTraceComparisonEntry;

  const evalId = value.currentRunValue?.evaluationId || value.otherRunValue?.evaluationId;

  // fetch colors if possible
  const currentRunUuid = value.currentRunValue?.traceInfo?.trace_metadata?.[MLFLOW_SOURCE_RUN_KEY];
  const otherRunUuid = value.otherRunValue?.traceInfo?.trace_metadata?.[MLFLOW_SOURCE_RUN_KEY];
  const currentRunColor = getRunColor && currentRunUuid ? getRunColor(currentRunUuid) : CURRENT_RUN_COLOR;
  const otherRunColor = getRunColor && otherRunUuid ? getRunColor(otherRunUuid) : COMPARE_TO_RUN_COLOR;

  const currentInputColumnTitle = value.currentRunValue
    ? getEvaluationResultInputTitle(value.currentRunValue, inputColumn)
    : undefined;

  const otherInputColumnTitle = value.otherRunValue
    ? getEvaluationResultInputTitle(value.otherRunValue, inputColumn)
    : undefined;

  const inputColumnTitle = currentInputColumnTitle || otherInputColumnTitle;
  const meta = row?.table?.options?.meta as
    | { getRunColor?: (runUuid: string) => string; searchQuery?: string }
    | undefined;
  const searchQuery = meta?.searchQuery;
  const displayContent =
    inputColumnTitle && searchQuery ? highlightSearchInText(String(inputColumnTitle), searchQuery) : inputColumnTitle;

  return (
    <div
      css={{
        height: '100%',
        width: '100%',
        display: 'flex',
        overflow: 'hidden',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: theme.spacing.sm,
      }}
    >
      <Typography.Link
        css={{
          display: 'block',
          overflow: 'hidden',
          whiteSpace: 'nowrap',
          textOverflow: 'ellipsis',
        }}
        componentId="mlflow.evaluations_review.table_ui.evaluation_id_link"
        onClick={() => onChangeEvaluationId(evalId, value.currentRunValue?.traceInfo)}
      >
        {displayContent ? (
          displayContent
        ) : (
          <Typography.Text
            color="secondary"
            css={{
              fontStyle: 'italic',
            }}
          >
            null
          </Typography.Text>
        )}
      </Typography.Link>
      {isComparing && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
          }}
        >
          <div
            css={{
              display: 'flex',
            }}
          >
            {currentRunUuid ? <RunColorCircle color={currentRunColor} /> : <div css={{ width: 12, height: 12 }} />}
          </div>
          <div
            css={{
              display: 'flex',
            }}
          >
            {otherRunUuid ? <RunColorCircle color={otherRunColor} /> : <div css={{ width: 12, height: 12 }} />}
          </div>
        </div>
      )}
    </div>
  );
};

export const traceInfoCellRenderer = (
  isComparing: boolean,
  colId: string,
  comparisonEntry: EvalTraceComparisonEntry,
  onChangeEvaluationId: (evalId: string, traceInfo?: ModelTraceInfoV3) => void,
  intl: IntlShape,
  theme: ThemeType,
  experimentId?: string,
  onTraceTagsEdit?: (trace: ModelTraceInfoV3) => void,
  traceIdToTurnMap?: Record<string, number>,
  searchQuery?: string,
) => {
  const currentTraceInfo = comparisonEntry.currentRunValue?.traceInfo;
  const otherTraceInfo = isComparing ? comparisonEntry.otherRunValue?.traceInfo : undefined;

  if (colId === REQUEST_TIME_COLUMN_ID) {
    const date = currentTraceInfo?.request_time ? new Date(currentTraceInfo.request_time) : undefined;
    const otherDate = otherTraceInfo?.request_time ? new Date(otherTraceInfo.request_time) : undefined;

    return (
      <StackedComponents
        first={
          date ? (
            <Tooltip
              componentId="mlflow.experiment-evaluation-monitoring.trace-info-hover-request-time"
              content={date.toLocaleString(navigator.language, { timeZoneName: 'short' })}
            >
              <span css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {formatDateTime(date, intl, {
                  year: 'numeric',
                  month: '2-digit',
                  day: '2-digit',
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                  hour12: false,
                })}
              </span>
            </Tooltip>
          ) : (
            <NullCell isComparing={isComparing} />
          )
        }
        second={
          isComparing &&
          (otherDate ? (
            <Tooltip
              componentId="mlflow.experiment-evaluation-monitoring.trace-info-hover-other-request-time"
              content={otherDate.toLocaleString(navigator.language, { timeZoneName: 'short' })}
            >
              <span css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{timeSinceStr(otherDate)}</span>
            </Tooltip>
          ) : (
            <NullCell isComparing={isComparing} />
          ))
        }
      />
    );
  } else if (colId === STATE_COLUMN_ID) {
    return (
      <StackedComponents
        first={<StatusCellRenderer original={currentTraceInfo} isComparing={isComparing} />}
        second={
          isComparing &&
          (otherTraceInfo ? (
            <StatusCellRenderer original={otherTraceInfo} isComparing={isComparing} />
          ) : (
            <NullCell isComparing={isComparing} />
          ))
        }
      />
    );
  } else if (colId === SOURCE_COLUMN_ID) {
    return (
      <StackedComponents
        first={
          currentTraceInfo ? (
            <SourceCellRenderer traceInfo={currentTraceInfo} isComparing={isComparing} />
          ) : (
            <NullCell isComparing={isComparing} />
          )
        }
        second={
          isComparing &&
          (otherTraceInfo ? (
            <SourceCellRenderer traceInfo={otherTraceInfo} isComparing={isComparing} />
          ) : (
            <NullCell isComparing={isComparing} />
          ))
        }
      />
    );
  } else if (colId === TAGS_COLUMN_ID) {
    const tagsArr: { key: string; value: string }[] = Object.entries(currentTraceInfo?.tags || {}).map(
      ([key, value]) => ({
        key,
        value,
      }),
    );

    const otherTagsArr: { key: string; value: string }[] = Object.entries(otherTraceInfo?.tags || {}).map(
      ([key, value]) => ({
        key,
        value,
      }),
    );

    // We only support editing tags in single trace mode
    const onAddEditTags = !otherTraceInfo && currentTraceInfo ? () => onTraceTagsEdit?.(currentTraceInfo) : undefined;

    return (
      <StackedComponents
        first={<TagsCellRenderer baseComponentId="tags-cell-renderer" tags={tagsArr} onAddEditTags={onAddEditTags} />}
        second={
          isComparing &&
          (otherTraceInfo ? (
            <TagsCellRenderer baseComponentId="tags-cell-renderer-other" tags={otherTagsArr} />
          ) : (
            <NullCell isComparing={isComparing} />
          ))
        }
      />
    );
  } else if (shouldEnableTagGrouping() && colId.startsWith(TAGS_COLUMN_ID)) {
    const tagKey = getTagKeyFromColumnId(colId);
    if (!tagKey) {
      return <NullCell isComparing={isComparing} />;
    }
    const tagValue = currentTraceInfo?.tags?.[tagKey];
    const otherTagValue = otherTraceInfo?.tags?.[tagKey];
    return (
      <StackedComponents
        first={
          tagValue ? (
            <span
              title={tagValue}
              css={{
                display: 'block',
                overflow: 'hidden',
                whiteSpace: 'nowrap',
                textOverflow: 'ellipsis',
              }}
            >
              {tagValue}
            </span>
          ) : (
            <NullCell isComparing={isComparing} />
          )
        }
        second={
          isComparing &&
          (otherTagValue ? (
            <span
              title={tagValue}
              css={{
                display: 'block',
                overflow: 'hidden',
                whiteSpace: 'nowrap',
                textOverflow: 'ellipsis',
              }}
            >
              {otherTagValue}
            </span>
          ) : (
            <NullCell isComparing={isComparing} />
          ))
        }
      />
    );
  } else if (colId === TRACE_NAME_COLUMN_ID) {
    const evalId = comparisonEntry.currentRunValue?.evaluationId;

    const currentTraceName = currentTraceInfo?.tags?.['mlflow.traceName'];
    const otherTraceName = otherTraceInfo?.tags?.['mlflow.traceName'];

    return (
      <StackedComponents
        first={
          currentTraceName ? (
            !isComparing ? (
              <Typography.Link
                css={{
                  display: 'block',
                  overflow: 'hidden',
                  whiteSpace: 'nowrap',
                  textOverflow: 'ellipsis',
                }}
                componentId="mlflow.evaluations_review.table_ui.evaluation_id_link"
                onClick={() => evalId && onChangeEvaluationId(evalId, comparisonEntry.currentRunValue?.traceInfo)}
              >
                {currentTraceInfo?.tags?.['mlflow.traceName']}
              </Typography.Link>
            ) : (
              <div>{currentTraceInfo?.tags?.['mlflow.traceName']}</div>
            )
          ) : (
            <NullCell isComparing={isComparing} />
          )
        }
        second={isComparing && (otherTraceName ? <div>{otherTraceName}</div> : <NullCell isComparing={isComparing} />)}
      />
    );
  } else if (colId === RUN_NAME_COLUMN_ID) {
    // This column is only shown on experiment level traces which does not support comparison mode

    const runUuid = currentTraceInfo?.trace_metadata?.[MLFLOW_SOURCE_RUN_KEY];

    if (!runUuid) {
      return <NullCell />;
    }

    return (
      <RunName
        experimentId={getExperimentIdFromTraceLocation(currentTraceInfo?.trace_location) ?? experimentId}
        runUuid={runUuid}
      />
    );
  } else if (colId === USER_COLUMN_ID) {
    const value = currentTraceInfo?.trace_metadata?.['mlflow.trace.user'] || currentTraceInfo?.tags?.['mlflow.user'];
    const otherValue = otherTraceInfo?.trace_metadata?.['mlflow.trace.user'] || otherTraceInfo?.tags?.['mlflow.user'];

    return (
      <StackedComponents
        first={
          value ? (
            <span
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
              }}
              title={value}
            >
              <UserIcon css={{ color: theme.colors.textSecondary, fontSize: 16 }} />
              <span css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{value}</span>
            </span>
          ) : (
            <NullCell isComparing={isComparing} />
          )
        }
        second={
          isComparing &&
          (otherValue ? (
            <span
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
              }}
              title={otherValue}
            >
              <UserIcon css={{ color: theme.colors.textSecondary, fontSize: 16 }} />
              <span css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{otherValue}</span>
            </span>
          ) : (
            <NullCell isComparing={isComparing} />
          ))
        }
      />
    );
  } else if (colId === TRACE_ID_COLUMN_ID) {
    // Regression-test view: display the test name (from the mlflow.test.* tags)
    // when present, but keep the real trace id for click/navigation. Falls back
    // to the trace id for ordinary runs, so this is a no-op outside test runs.
    const testDisplayName = (info?: ModelTraceInfoV3): string | undefined => {
      const name = readTraceTag(info, 'mlflow.test.name');
      if (!name) return undefined;
      const caseId = readTraceTag(info, 'mlflow.test.case_id');
      return caseId ? `${name}[${caseId}]` : name;
    };
    const navId = currentTraceInfo?.trace_id;
    const otherNavId = otherTraceInfo?.trace_id;
    const testName = testDisplayName(currentTraceInfo);
    const otherTestName = testDisplayName(otherTraceInfo);
    const value = testName ?? navId;
    const otherValue = otherTestName ?? otherNavId;
    const displayValue = value && searchQuery ? highlightSearchInText(value, searchQuery) : value;
    const displayOtherValue = otherValue && searchQuery ? highlightSearchInText(otherValue, searchQuery) : otherValue;
    return (
      <StackedComponents
        first={
          value ? (
            <Tag
              css={{ width: 'fit-content', maxWidth: '100%' }}
              componentId="mlflow.genai-traces-table.trace-id"
              color={testName ? 'purple' : 'indigo'}
              title={value}
              onClick={() => navId && onChangeEvaluationId(navId, currentTraceInfo)}
            >
              <span
                css={{
                  display: 'block',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {displayValue}
              </span>
            </Tag>
          ) : (
            <NullCell isComparing={isComparing} />
          )
        }
        second={
          isComparing &&
          (otherValue ? (
            <Tag
              css={{ width: 'fit-content', maxWidth: '100%' }}
              componentId="mlflow.genai-traces-table.trace-id"
              color={otherTestName ? 'purple' : 'indigo'}
              title={otherValue}
              onClick={() => otherNavId && onChangeEvaluationId(otherNavId, otherTraceInfo)}
            >
              <span
                css={{
                  display: 'inline-block',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {displayOtherValue}
              </span>
            </Tag>
          ) : (
            <NullCell isComparing={isComparing} />
          ))
        }
      />
    );
  } else if (colId === SESSION_COLUMN_ID) {
    const value = currentTraceInfo?.trace_metadata?.['mlflow.trace.session'];
    const otherValue = otherTraceInfo?.trace_metadata?.['mlflow.trace.session'];
    const currentTraceId = currentTraceInfo?.trace_id;
    const otherTraceId = otherTraceInfo?.trace_id;

    const turnNumber = traceIdToTurnMap?.[currentTraceId ?? ''];
    const otherTurnNumber = traceIdToTurnMap?.[otherTraceId ?? ''];

    return (
      <StackedComponents
        first={
          value ? (
            <SessionIdLinkWrapper
              sessionId={value}
              experimentId={getExperimentIdFromTraceLocation(currentTraceInfo?.trace_location) ?? experimentId}
              traceId={currentTraceId}
            >
              <Tag
                css={{ width: 'fit-content', maxWidth: '100%' }}
                componentId="mlflow.genai-traces-table.session"
                title={value}
              >
                <span
                  css={{
                    display: 'inline-block',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {!isNil(turnNumber) ? (
                    <FormattedMessage
                      defaultMessage="Turn {turnNumber}"
                      description="Label for a single turn within an experiment chat session"
                      values={{ turnNumber }}
                    />
                  ) : (
                    value
                  )}
                </span>
              </Tag>
            </SessionIdLinkWrapper>
          ) : (
            <NullCell isComparing={isComparing} />
          )
        }
        second={
          isComparing &&
          (otherValue ? (
            <SessionIdLinkWrapper
              sessionId={otherValue}
              experimentId={getExperimentIdFromTraceLocation(otherTraceInfo?.trace_location) ?? experimentId}
              traceId={otherTraceId}
            >
              <Tag
                css={{ width: 'fit-content', maxWidth: '100%' }}
                componentId="mlflow.genai-traces-table.session"
                title={otherValue}
              >
                <span
                  css={{
                    display: 'inline-block',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {!isNil(otherTurnNumber) ? (
                    <FormattedMessage
                      defaultMessage="Turn {turnNumber}"
                      description="Label for a single turn within an experiment chat session"
                      values={{ turnNumber: otherTurnNumber }}
                    />
                  ) : (
                    value
                  )}
                </span>
              </Tag>
            </SessionIdLinkWrapper>
          ) : (
            <NullCell isComparing={isComparing} />
          ))
        }
      />
    );
  } else if (colId === RESPONSE_COLUMN_ID) {
    const value = currentTraceInfo ? formatResponseTitle(getTraceInfoOutputs(currentTraceInfo)) : '';
    const otherValue = otherTraceInfo ? formatResponseTitle(getTraceInfoOutputs(otherTraceInfo)) : '';
    const displayValue = value && searchQuery ? highlightSearchInText(value, searchQuery) : value;
    const displayOtherValue = otherValue && searchQuery ? highlightSearchInText(otherValue, searchQuery) : otherValue;
    return (
      <StackedComponents
        first={
          displayValue ? (
            <div css={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={value}>
              {displayValue}
            </div>
          ) : (
            <Typography.Text
              color="secondary"
              css={{
                fontStyle: 'italic',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              null
            </Typography.Text>
          )
        }
        second={
          isComparing &&
          (displayOtherValue ? (
            <div css={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={otherValue}>
              {displayOtherValue}
            </div>
          ) : (
            <Typography.Text
              color="secondary"
              css={{
                fontStyle: 'italic',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              null
            </Typography.Text>
          ))
        }
      />
    );
  } else if (colId === LOGGED_MODEL_COLUMN_ID) {
    return (
      <LoggedModelCell
        experimentId={experimentId}
        currentTraceInfo={currentTraceInfo}
        otherTraceInfo={otherTraceInfo}
        isComparing={isComparing}
      />
    );
  } else if (colId === TOKENS_COLUMN_ID) {
    return <TokensCell currentTraceInfo={currentTraceInfo} otherTraceInfo={otherTraceInfo} isComparing={isComparing} />;
  } else if (colId === ISSUES_COLUMN_ID) {
    const issues = comparisonEntry.currentRunValue?.issues;
    const otherIssues = comparisonEntry.otherRunValue?.issues;
    return <IssuesCell issues={issues} otherIssues={otherIssues} isComparing={isComparing} />;
  } else if (colId.startsWith(CUSTOM_METADATA_COLUMN_ID)) {
    const metadataKey = getCustomMetadataKeyFromColumnId(colId);
    if (!metadataKey) {
      return <NullCell isComparing={isComparing} />;
    }
    const value = currentTraceInfo?.trace_metadata?.[metadataKey];
    const otherValue = otherTraceInfo?.trace_metadata?.[metadataKey];
    return (
      <StackedComponents
        first={
          value ? (
            <div css={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={value}>
              {value}
            </div>
          ) : (
            <NullCell isComparing={isComparing} />
          )
        }
        second={
          isComparing &&
          (otherValue ? (
            <div css={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={otherValue}>
              {otherValue}
            </div>
          ) : (
            <NullCell isComparing={isComparing} />
          ))
        }
      />
    );
  } else if (colId === EXECUTION_DURATION_COLUMN_ID) {
    const value = normalizeDurationString(currentTraceInfo?.[EXECUTION_DURATION_COLUMN_ID]);
    const otherValue = normalizeDurationString(otherTraceInfo?.[EXECUTION_DURATION_COLUMN_ID]);

    return (
      <StackedComponents
        first={!isNil(value) ? <ExecutionDurationTag value={value} /> : <NullCell isComparing={isComparing} />}
        second={
          isComparing &&
          (!isNil(otherValue) ? <ExecutionDurationTag value={otherValue} /> : <NullCell isComparing={isComparing} />)
        }
      />
    );
  } else if (colId === LINKED_PROMPTS_COLUMN_ID) {
    const PROMPT_VERSION_QUERY_PARAM = 'promptVersion';
    const PROMPT_PATH = '/experiments/:experimentId/prompts/:promptName';

    const formatPrompts = (promptsJson: string | undefined, traceExperimentId: string | undefined) => {
      if (!promptsJson) return null;
      try {
        const prompts = JSON.parse(promptsJson);
        if (Array.isArray(prompts) && prompts.length > 0) {
          return prompts.map((prompt: { name: string; version: string }, index: number) => {
            const label = `${prompt.name}/${prompt.version}`;
            if (traceExperimentId) {
              const basePath = generatePath(PROMPT_PATH, { experimentId: traceExperimentId, promptName: prompt.name });
              const url = prompt.version
                ? `${basePath}?${new URLSearchParams({ [PROMPT_VERSION_QUERY_PARAM]: prompt.version }).toString()}`
                : basePath;
              return (
                <div key={index}>
                  <Link componentId="mlflow.genai-traces-table.prompt_link" to={url}>
                    {label}
                  </Link>
                </div>
              );
            }
            return <div key={index}>{label}</div>;
          });
        }
      } catch (e) {
        // Invalid JSON, return as-is
        return promptsJson;
      }
      return null;
    };

    const currentPrompts = currentTraceInfo?.tags?.['mlflow.linkedPrompts'];
    const otherPrompts = otherTraceInfo?.tags?.['mlflow.linkedPrompts'];
    const currentExperimentId = getExperimentIdFromTraceLocation(currentTraceInfo?.trace_location) ?? experimentId;
    const otherExperimentId = getExperimentIdFromTraceLocation(otherTraceInfo?.trace_location) ?? experimentId;
    const formattedCurrentPrompts = formatPrompts(currentPrompts, currentExperimentId);
    const formattedOtherPrompts = formatPrompts(otherPrompts, otherExperimentId);

    return (
      <StackedComponents
        first={
          formattedCurrentPrompts ? (
            <div css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{formattedCurrentPrompts}</div>
          ) : (
            <NullCell isComparing={isComparing} />
          )
        }
        second={
          isComparing &&
          (formattedOtherPrompts ? (
            <div css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{formattedOtherPrompts}</div>
          ) : (
            <NullCell isComparing={isComparing} />
          ))
        }
      />
    );
  } else if (colId === SIMULATION_GOAL_COLUMN_ID || colId === SIMULATION_PERSONA_COLUMN_ID) {
    // Goal/Persona are session-level columns, only rendered in session header rows
    return <NullCell isComparing={isComparing} />;
  }

  const value = currentTraceInfo ? stringifyValue(getTraceInfoValueWithColId(currentTraceInfo, colId)) : '';
  const otherValue = otherTraceInfo ? stringifyValue(getTraceInfoValueWithColId(otherTraceInfo, colId)) : '';

  return (
    <StackedComponents
      first={
        value ? (
          <div css={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={value}>
            {value}
          </div>
        ) : (
          <NullCell isComparing={isComparing} />
        )
      }
      second={
        isComparing &&
        (otherValue ? (
          <div css={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={otherValue}>
            {otherValue}
          </div>
        ) : (
          <NullCell isComparing={isComparing} />
        ))
      }
    />
  );
};

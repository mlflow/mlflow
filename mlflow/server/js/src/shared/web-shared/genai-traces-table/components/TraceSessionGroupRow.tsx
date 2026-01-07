import React from 'react';
import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  Typography,
  useDesignSystemTheme,
  CheckCircleFillIcon,
  XCircleFillIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { AssessmentInfo, SessionAssessmentAggregate, TraceSessionGroup } from '../types';
import MlflowUtils from '../utils/MlflowUtils';

export interface TraceSessionGroupRowProps {
  sessionGroup: TraceSessionGroup;
  isExpanded: boolean;
  onToggleExpand: () => void;
  selectedAssessmentInfos: AssessmentInfo[];
}

interface SessionAssessmentCellProps {
  aggregate: SessionAssessmentAggregate | undefined;
  assessmentInfo: AssessmentInfo;
}

const SessionAssessmentCell = ({ aggregate, assessmentInfo }: SessionAssessmentCellProps) => {
  const { theme } = useDesignSystemTheme();

  if (!aggregate || aggregate.totalCount === 0) {
    return <span css={{ color: theme.colors.textSecondary }}>-</span>;
  }

  const { dtype } = assessmentInfo;

  // For numeric assessments, show average
  if (dtype === 'numeric' && aggregate.numericAverage !== null) {
    return (
      <span css={{ color: theme.colors.textPrimary }}>
        {aggregate.numericAverage.toFixed(2)}
        <span css={{ color: theme.colors.textSecondary, marginLeft: theme.spacing.xs }}>avg</span>
      </span>
    );
  }

  // For pass-fail and boolean, show pass count
  if (dtype === 'pass-fail' || dtype === 'boolean') {
    const isPassing = aggregate.sessionPassed === true;
    const icon = isPassing ? (
      <CheckCircleFillIcon css={{ color: theme.colors.textValidationSuccess }} />
    ) : (
      <XCircleFillIcon css={{ color: theme.colors.textValidationDanger }} />
    );

    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        {icon}
        <span>
          {aggregate.passCount}/{aggregate.totalCount}
        </span>
      </div>
    );
  }

  // For string or unknown types
  return <span css={{ color: theme.colors.textSecondary }}>-</span>;
};

export const TraceSessionGroupRow = React.memo(
  ({ sessionGroup, isExpanded, onToggleExpand, selectedAssessmentInfos }: TraceSessionGroupRowProps) => {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();

    const formattedStartTime = sessionGroup.sessionStartTime
      ? MlflowUtils.formatTimestamp(new Date(sessionGroup.sessionStartTime))
      : null;

    const traceCountLabel = intl.formatMessage(
      {
        defaultMessage: '{count, plural, one {# trace} other {# traces}}',
        description: 'Label showing number of traces in a session',
      },
      { count: sessionGroup.traceCount },
    );

    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          backgroundColor: theme.colors.backgroundSecondary,
          borderBottom: `1px solid ${theme.colors.border}`,
          gap: theme.spacing.md,
          minHeight: 48,
        }}
      >
        {/* Expand/Collapse Button */}
        <Button
          componentId="mlflow.traces-table.session-expand-button"
          size="small"
          css={{ flexShrink: 0 }}
          icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={onToggleExpand}
        />

        {/* Session Info */}
        <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Session:" description="Label for session group row in traces table" />
            </Typography.Text>
            <Typography.Text
              css={{
                maxWidth: 200,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              title={sessionGroup.sessionId}
            >
              {sessionGroup.sessionId}
            </Typography.Text>
          </div>
          <Typography.Text size="sm" color="secondary">
            {traceCountLabel}
            {formattedStartTime && ` · Started ${formattedStartTime}`}
          </Typography.Text>
        </div>

        {/* Assessment Aggregates */}
        <div css={{ display: 'flex', gap: theme.spacing.lg, alignItems: 'center' }}>
          {selectedAssessmentInfos.map((assessmentInfo) => {
            const aggregate = sessionGroup.aggregatedAssessments.get(assessmentInfo.name);
            return (
              <div
                key={assessmentInfo.name}
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  minWidth: 60,
                }}
              >
                <Typography.Text
                  size="sm"
                  color="secondary"
                  css={{
                    maxWidth: 80,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={assessmentInfo.displayName}
                >
                  {assessmentInfo.displayName}
                </Typography.Text>
                <SessionAssessmentCell aggregate={aggregate} assessmentInfo={assessmentInfo} />
              </div>
            );
          })}
        </div>
      </div>
    );
  },
);

TraceSessionGroupRow.displayName = 'TraceSessionGroupRow';

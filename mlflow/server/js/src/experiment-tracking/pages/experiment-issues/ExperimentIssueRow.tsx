import {
  CheckCircleIcon,
  PreviewCard,
  Typography,
  VisibleOffIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import Utils from '../../../common/utils/Utils';
import { SEVERITY_TAG_CONFIG } from '../../components/run-page/IssueCard';
import type { Issue, IssueSeverity, IssueStatus } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { IssueCategoryTag } from './IssueCategoryTag';
import { STATUS_LABELS } from './issueLabels';

const IssueSeverityGlyph = ({ severity }: { severity: IssueSeverity }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const filledBars = { high: 3, medium: 2, low: 1, not_an_issue: 0 }[severity];
  const color =
    severity === 'high'
      ? theme.colors.textValidationDanger
      : severity === 'medium'
        ? theme.colors.textValidationWarning
        : theme.colors.textSecondary;
  return (
    <svg
      width={12}
      height={11}
      viewBox="0 0 12 11"
      role="img"
      aria-label={intl.formatMessage(
        { defaultMessage: 'Severity: {severity}', description: 'Experiment Issue severity glyph accessible label' },
        { severity: intl.formatMessage(SEVERITY_TAG_CONFIG[severity].label) },
      )}
      css={{ display: 'block', alignSelf: 'center' }}
    >
      {Array.from({ length: 3 }, (_, index) => (
        <rect
          key={index}
          x={index * 4.5}
          y={7 - index * 3}
          width={3}
          height={4 + index * 3}
          rx={0.5}
          fill={index < filledBars ? color : theme.colors.borderDecorative}
        />
      ))}
    </svg>
  );
};

export const IssueStatusMarker = ({ status }: { status: IssueStatus }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <span
      aria-hidden
      css={{
        width: 7,
        height: 7,
        flexShrink: 0,
        borderRadius: theme.borders.borderRadiusFull,
        backgroundColor: {
          pending: theme.colors.textValidationDanger,
          resolved: theme.colors.textValidationSuccess,
          rejected: theme.colors.textSecondary,
        }[status],
      }}
    />
  );
};

export const IssueCompletionIcon = ({ status }: { status: IssueStatus }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  if (status === 'pending') return null;
  return (
    <span
      role="img"
      aria-label={intl.formatMessage(STATUS_LABELS[status])}
      css={{
        display: 'inline-flex',
        flexShrink: 0,
        color: status === 'resolved' ? theme.colors.textValidationSuccess : theme.colors.textSecondary,
      }}
    >
      {status === 'resolved' ? <CheckCircleIcon aria-hidden /> : <VisibleOffIcon aria-hidden />}
    </span>
  );
};

export const ExperimentIssueRow = ({
  issue,
  isSelected,
  onSelect,
}: {
  issue: Issue;
  isSelected: boolean;
  onSelect: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <PreviewCard
      componentId="mlflow.experiment-issues.row"
      selected={isSelected}
      onClick={onSelect}
      icon={
        issue.status !== 'pending' ? (
          <IssueCompletionIcon status={issue.status} />
        ) : issue.severity ? (
          <IssueSeverityGlyph severity={issue.severity} />
        ) : undefined
      }
      title={
        <Typography.Text bold ellipsis title={issue.name} css={{ maxWidth: '100%' }}>
          {issue.name}
        </Typography.Text>
      }
      titleActions={
        <Typography.Hint css={{ whiteSpace: 'nowrap' }}>{Utils.timeSinceStr(issue.created_timestamp)}</Typography.Hint>
      }
      dangerouslyAppendEmotionCSS={{
        width: '100%',
        padding: theme.spacing.md,
        gap: theme.spacing.xs,
        borderRadius: 0,
        boxShadow: 'none',
        border: 'none',
        borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        borderLeft: `2px solid ${isSelected ? theme.colors.actionPrimaryBackgroundDefault : 'transparent'}`,
        background: isSelected ? theme.colors.actionTertiaryBackgroundPress : 'transparent',
        '&:hover': {
          background: isSelected ? undefined : theme.colors.actionDefaultBackgroundHover,
          borderColor: 'transparent',
          borderBottomColor: theme.colors.borderDecorative,
          borderLeftColor: isSelected ? theme.colors.actionPrimaryBackgroundDefault : 'transparent',
          boxShadow: 'none',
        },
        // Flat rows carry selection on the leading edge; reserve the outline for keyboard focus.
        '&[aria-pressed="true"]': {
          outline: 'none',
          boxShadow: 'none',
          borderBottomColor: theme.colors.borderDecorative,
          borderLeftColor: theme.colors.actionPrimaryBackgroundDefault,
        },
        '&:focus': { outline: 'none', boxShadow: 'none' },
        '&:focus-visible': {
          outline: `2px solid ${theme.colors.actionDefaultBorderFocus}`,
          outlineOffset: -2,
          boxShadow: 'none',
        },
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: theme.spacing.sm,
          marginInlineStart: theme.spacing.lg,
          pointerEvents: 'none',
        }}
      >
        {issue.categories?.map((category) => (
          <IssueCategoryTag
            key={category}
            category={category}
            componentId="mlflow.experiment-issues.row.category"
            insideClickableRow
          />
        ))}
        {issue.trace_count !== undefined && (
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="{count, plural, one {# trace} other {# traces}}"
              description="Experiment Issue affected trace count"
              values={{ count: issue.trace_count }}
            />
          </Typography.Text>
        )}
      </div>
    </PreviewCard>
  );
};

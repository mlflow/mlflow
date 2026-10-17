import {
  CheckCircleIcon,
  PreviewCard,
  Typography,
  useDesignSystemTheme,
  VisibleOffIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import Utils from '../../../common/utils/Utils';
import type { Issue, IssueSeverity, IssueStatus } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { IssueCategoryTag } from './IssueCategoryTag';
import { SEVERITY_GLYPH_LABELS, STATUS_LABELS } from './issueLabels';

// Severity is the list's primary scan key, so it gets a glyph rather than a text tag: three
// signal bars, filled by rank. Bar count carries the ordering on its own, so severity survives
// without the colour (WCAG 1.4.1), and the glyph costs a fraction of a "Medium" tag's width.
const SEVERITY_GLYPH_BARS = 3;
const SEVERITY_FILLED_BARS: Record<IssueSeverity, number> = { high: 3, medium: 2, low: 1, not_an_issue: 0 };

export const IssueSeverityGlyph = ({ severity }: { severity: IssueSeverity }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const filledColor = {
    high: theme.colors.textValidationDanger,
    medium: theme.colors.textValidationWarning,
    low: theme.colors.textSecondary,
    not_an_issue: theme.colors.textSecondary,
  }[severity];
  const filledBars = SEVERITY_FILLED_BARS[severity];

  return (
    <svg
      width={12}
      height={11}
      viewBox="0 0 12 11"
      role="img"
      aria-label={intl.formatMessage(SEVERITY_GLYPH_LABELS[severity])}
      // Centred in its cell, whose height is the title's line box, so the glyph reads as
      // sitting on the name rather than floating above it.
      css={{ display: 'block', alignSelf: 'center' }}
    >
      {Array.from({ length: SEVERITY_GLYPH_BARS }, (_, index) => {
        const height = 4 + index * 3;
        return (
          <rect
            key={index}
            x={index * 4.5}
            y={11 - height}
            width={3}
            height={height}
            rx={0.5}
            fill={index < filledBars ? filledColor : theme.colors.borderDecorative}
          />
        );
      })}
    </svg>
  );
};

export const IssueStatusMarker = ({ status, compact = false }: { status: IssueStatus; compact?: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const backgroundColor = {
    pending: theme.colors.textValidationDanger,
    resolved: theme.colors.textValidationSuccess,
    rejected: theme.colors.textSecondary,
  }[status];

  return (
    <span
      aria-hidden
      css={{
        width: compact ? 6 : 7,
        height: compact ? 6 : 7,
        flexShrink: 0,
        borderRadius: theme.borders.borderRadiusFull,
        backgroundColor,
      }}
    />
  );
};

export const IssueCompletionIcon = ({ status }: { status: IssueStatus }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  if (status === 'resolved') {
    return (
      <span
        role="img"
        aria-label={intl.formatMessage(STATUS_LABELS.resolved)}
        css={{ display: 'inline-flex', color: theme.colors.textValidationSuccess, flexShrink: 0 }}
      >
        <CheckCircleIcon aria-hidden />
      </span>
    );
  }
  if (status === 'rejected') {
    return (
      <span
        role="img"
        aria-label={intl.formatMessage(STATUS_LABELS.rejected)}
        css={{ display: 'inline-flex', color: theme.colors.textSecondary, flexShrink: 0 }}
      >
        <VisibleOffIcon aria-hidden />
      </span>
    );
  }
  return null;
};

export interface ExperimentIssueRowProps {
  issue: Issue;
  isSelected: boolean;
  onSelect: () => void;
  /**
   * The trace count to show, or null to show none. Defaults to the exact SQL trace count.
   */
  affectedTraceCount?: number | null;
}

/**
 * One issue as a single scannable row: severity, name, categories, blast radius and age.
 *
 * Built on `PreviewCard`, which is the only dbui control that is simultaneously interactive
 * (`role="button"`, `tabIndex`, Enter/Space), selectable from *outside* state via `selected`
 * (this tab's selection lives in the URL), and restyleable without `!important` -- its
 * `dangerouslyAppendEmotionCSS` is applied after the container styles, so the flush 61px row
 * below wins on ordering. Its header/children split maps onto the two lines directly.
 *
 * The alternatives, for the next person who asks: `Button` bakes height/display/radius behind
 * `!important` and wraps children in a span, so a two-line grid needs `!important` to land at
 * all; `Listbox` is the ideal semantic but its selection is internal `useState` with no `value`
 * prop, which cannot be driven from the URL; `Card` is a roleless div, so `aria-pressed` is
 * inert on it -- the reason the retired IssueCard's selected state was invisible to assistive
 * tech; `RadioTile` takes no children.
 *
 * The description is deliberately absent -- a list of full descriptions fits about ten issues in
 * a viewport, and the detail pane is where an issue is read.
 */
export const ExperimentIssueRow = ({
  issue,
  isSelected,
  onSelect,
  affectedTraceCount = issue.trace_count ?? null,
}: ExperimentIssueRowProps) => {
  const { theme } = useDesignSystemTheme();
  const age = Utils.timeSinceStr(issue.created_timestamp);
  return (
    <PreviewCard
      componentId="mlflow.experiment-issues.row"
      data-testid="experiment-issue-row"
      onClick={onSelect}
      // PreviewCard renders role="button" and aria-pressed={selected}, so selection stays driven
      // by the URL rather than by internal component state.
      selected={isSelected}
      icon={
        issue.status === 'pending' ? (
          issue.severity ? (
            <IssueSeverityGlyph severity={issue.severity} />
          ) : undefined
        ) : (
          <IssueCompletionIcon status={issue.status} />
        )
      }
      title={
        <Typography.Text bold ellipsis title={issue.name}>
          {issue.name}
        </Typography.Text>
      }
      titleActions={<Typography.Hint css={{ whiteSpace: 'nowrap' }}>{age}</Typography.Hint>}
      // The card's own chrome is padding, a full border, a radius and a shadow per row; a flush
      // 61px list row undoes those. dangerouslyAppendEmotionCSS is applied after the container
      // styles, so this wins on ordering without needing !important.
      dangerouslyAppendEmotionCSS={{
        width: '100%',
        padding: `${theme.spacing.md}px`,
        gap: theme.spacing.xs,
        borderRadius: 0,
        boxShadow: 'none',
        border: 'none',
        borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        // Marked on the leading edge so rows do not shift as the selection moves down the list.
        borderLeft: `2px solid ${isSelected ? theme.colors.actionPrimaryBackgroundDefault : 'transparent'}`,
        background: isSelected ? theme.colors.actionTertiaryBackgroundPress : 'transparent',
        '&:hover': {
          background: isSelected ? undefined : theme.colors.actionDefaultBackgroundHover,
          borderColor: 'transparent',
          borderBottomColor: theme.colors.borderDecorative,
          borderLeftColor: isSelected ? theme.colors.actionPrimaryBackgroundDefault : 'transparent',
          boxShadow: 'none',
        },
        // PreviewCard treats a selected card as an outlined, shadowed, accent-bordered box, and
        // its `&[aria-pressed="true"]` rule outranks the flat declarations above. In a flush list
        // that reads as a focus ring on a row nobody is focused on, so it is undone here -- the
        // selection is already carried by the tint and the leading edge.
        '&[aria-pressed="true"]': {
          outline: 'none',
          boxShadow: 'none',
          borderBottomColor: theme.colors.borderDecorative,
          borderLeftColor: theme.colors.actionPrimaryBackgroundDefault,
        },
        // Re-asserted after the rule above so a keyboard-focused row keeps a ring even when it is
        // the selected one.
        // PreviewCard rings and shadows on plain `:focus`, which a div with tabIndex picks up on
        // mouse-down too. Drop both, then re-add the ring for keyboard focus only, so clicking a
        // row does not leave it outlined and no row is ever lifted out of a flush list.
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
          // Tag's wrapper carries tabIndex={-1} even when it is not clickable, so a click reaching
          // it would focus that div instead of the row. The other half of this lives in
          // IssueCategoryTag: `insideClickableRow` re-enables pointer events on its tooltip
          // trigger alone, so hover works and the click still arrives here.
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
        {affectedTraceCount !== null && (
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="{count, plural, one {# trace} other {# traces}}"
              description="Experiment issues tab > List row > Number of traces the issue affects"
              values={{ count: affectedTraceCount }}
            />
          </Typography.Text>
        )}
      </div>
    </PreviewCard>
  );
};

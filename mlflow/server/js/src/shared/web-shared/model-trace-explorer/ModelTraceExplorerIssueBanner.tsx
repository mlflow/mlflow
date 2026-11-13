import {
  Button,
  DangerIcon,
  NewWindowIcon,
  SparkleIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';

import { type TraceIssueReference } from './IssueAssessmentUtils';
import { useVisibleIssueReferences } from './hooks/useVisibleIssueReferences';
import type { ModelTraceSpanNode } from './ModelTrace.types';
import { Link } from './RoutingUtils';

/**
 * The Issues tab reads its selection from this query parameter, so an issue named on a span links
 * straight to its detail pane. Spelled out here because web-shared cannot reach the page's routes.
 */
const SELECTED_ISSUE_ID_PARAM = 'selectedIssueId';

export const getIssueDetailsHref = (experimentId: string, issue: TraceIssueReference): string => {
  const searchParams = new URLSearchParams({ [SELECTED_ISSUE_ID_PARAM]: issue.id });
  return `/experiments/${encodeURIComponent(experimentId)}/issues?${searchParams}`;
};

/**
 * The issues detection flagged on the active span, as cards above its inputs and outputs.
 *
 * A span the Issues tab blamed is where a reader lands when they follow a piece of evidence, so
 * the trace itself has to say which issue brought them there and why, rather than leaving that
 * context behind in the pane they came from.
 */
export const ModelTraceExplorerIssueEvidence = ({
  activeSpan,
  experimentId,
}: {
  activeSpan: ModelTraceSpanNode;
  experimentId?: string;
}): JSX.Element | null => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const viewIssueDetailsLabel = intl.formatMessage({
    defaultMessage: 'Open issue details in a new tab',
    description: 'Tooltip for opening the issue details page in a new browser tab',
  });
  const issues = useVisibleIssueReferences(activeSpan.assessments);
  if (issues.length === 0) {
    return null;
  }

  const backgroundColor = theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        marginTop: theme.spacing.sm,
        marginBottom: theme.spacing.sm,
      }}
    >
      {issues.map((issue) => {
        const issueDetailsHref = experimentId ? getIssueDetailsHref(experimentId, issue) : undefined;
        return (
          <div
            key={issue.id}
            role="status"
            data-component-id="shared.model-trace-explorer.span-issue-evidence"
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              padding: theme.spacing.sm,
              border: '1px solid transparent',
              borderRadius: theme.borders.borderRadiusSm,
              background: `linear-gradient(${backgroundColor}, ${backgroundColor}) padding-box, ${theme.gradients.aiBorderGradient} border-box`,
            }}
          >
            <SparkleIcon color="ai" css={{ flexShrink: 0 }} />
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, minWidth: 0, flex: 1 }}>
              <Typography.Text bold>{issue.name}</Typography.Text>
              {issue.rationale && <Typography.Text color="secondary">{issue.rationale}</Typography.Text>}
            </div>
            {issueDetailsHref && (
              <Link
                componentId="shared.model-trace-explorer.span-issue-evidence.view-issue-details"
                to={issueDetailsHref}
                target="_blank"
                rel="noreferrer"
              >
                <Tooltip
                  componentId="shared.model-trace-explorer.span-issue-evidence.view-issue-details-tooltip"
                  content={viewIssueDetailsLabel}
                >
                  <Button
                    aria-label={viewIssueDetailsLabel}
                    componentId="shared.model-trace-explorer.span-issue-evidence.view-issue-details-button"
                    icon={<NewWindowIcon />}
                    size="small"
                    type="tertiary"
                    css={{ flexShrink: 0 }}
                  />
                </Tooltip>
              </Link>
            )}
          </div>
        );
      })}
    </div>
  );
};

/**
 * How many issues a span is flagged for, on its row in the span tree, so the offending span is
 * findable without opening every span in the trace.
 */
export const ModelTraceExplorerIssueIndicator = ({ span }: { span: ModelTraceSpanNode }): JSX.Element | null => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const issues = useVisibleIssueReferences(span.assessments);
  if (issues.length === 0) {
    return null;
  }

  const label = intl.formatMessage(
    {
      defaultMessage: '{count, plural, one {Issue detected: {issueName}} other {# issues detected}}',
      description: 'Tooltip for an issue indicator on a trace span tree node',
    },
    { count: issues.length, issueName: issues[0]?.name },
  );
  return (
    <Tooltip content={label} componentId="shared.model-trace-explorer.span-issue-indicator-tooltip">
      <Tag
        aria-label={label}
        color="purple"
        componentId="shared.model-trace-explorer.span-issue-indicator"
        data-testid={`span-issue-indicator-${span.key}`}
        css={{
          margin: 0,
          marginLeft: theme.spacing.xs,
          borderRadius: theme.borders.borderRadiusSm,
          flexShrink: 0,
        }}
      >
        <DangerIcon css={{ fontSize: theme.typography.fontSizeSm }} />
        <Typography.Text css={{ marginLeft: theme.spacing.xs }}>{issues.length}</Typography.Text>
      </Tag>
    </Tooltip>
  );
};

import {
  Button,
  CheckIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  CopyIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCopyController } from '@databricks/web-shared/copy';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';

import { useLocation } from '../../../common/utils/RoutingUtils';
import type { Issue } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { IssueDetailSectionTitle } from './IssueDetailSectionTitle';
import { buildIssueFixPrompt } from './issueFixPrompt';
import { useIssueEvidenceTraces } from './useIssueEvidenceTraces';

/** How many lines of the prompt show until it is expanded, and their height in `em`. */
const COLLAPSED_PROMPT_LINES = 5;
const PROMPT_LINE_HEIGHT = 1.6;

export interface IssueProposedFixSectionProps {
  experimentId: string;
  issue: Issue;
}

/**
 * The fix prompt for an issue, for a coding agent to run in the connected repository. It is
 * collapsed to its first lines, and its Copy button stays visible either way.
 *
 * The affected traces come from the same query the Evidence section uses, so naming them in the
 * prompt costs no extra request.
 */
export const IssueProposedFixSection = ({ experimentId, issue }: IssueProposedFixSectionProps) => {
  const { theme } = useDesignSystemTheme();
  // Derived from the issue rather than generated: one fix section renders per issue, and issue IDs
  // are already constrained to characters an HTML id accepts.
  const headingId = `issue-fix-heading-${issue.issue_id}`;
  const promptId = `issue-fix-prompt-${issue.issue_id}`;
  const [isExpanded, setIsExpanded] = useState(false);
  const { pathname, search, hash } = useLocation();
  // Absolute, and hash-routed, so the link in the agent's pull request opens from anywhere.
  const issueUrl = `${window.location.origin}/#${pathname}${search}${hash}`;
  const { traceInfos } = useIssueEvidenceTraces(experimentId, issue.issue_id);
  const traceIds = useMemo(
    () => traceInfos.map((trace) => trace.trace_id).filter((traceId): traceId is string => Boolean(traceId)),
    [traceInfos],
  );
  const prompt = buildIssueFixPrompt(experimentId, issue, issueUrl, {
    traceIds,
    totalTraceCount: issue.trace_count,
  });
  const { copied, copy } = useCopyController(prompt);

  return (
    <section
      aria-labelledby={headingId}
      data-testid="experiment-issue-proposed-fix"
      css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <IssueDetailSectionTitle id={headingId}>
          <FormattedMessage
            defaultMessage="Fix this issue"
            description="Experiment issues tab > Issue detail > Heading of the section with a prompt that fixes the issue"
          />
        </IssueDetailSectionTitle>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Copy the prompt below to fix this issue with your coding agent."
            description="Experiment issues tab > Issue detail > Fix this issue section > Explains what the prompt below is for"
          />
        </Typography.Text>
      </div>
      <div
        css={{
          position: 'relative',
          overflow: 'hidden',
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          backgroundColor: theme.colors.backgroundSecondary,
        }}
      >
        <div css={{ position: 'absolute', top: theme.spacing.sm, right: theme.spacing.sm, zIndex: 1 }}>
          <Button
            componentId="mlflow.experiment-issues.detail.proposed-fix.copy"
            size="small"
            icon={copied ? <CheckIcon /> : <CopyIcon />}
            onClick={copy}
          >
            {copied ? (
              <FormattedMessage
                defaultMessage="Copied"
                description="Experiment issues tab > Issue detail > Fix this issue > Shown after the fix prompt was copied"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Copy"
                description="Experiment issues tab > Issue detail > Fix this issue > Copies the fix prompt"
              />
            )}
          </Button>
        </div>
        <pre
          id={promptId}
          css={{
            margin: 0,
            // Room on the right, so the first lines don't run under the Copy button.
            padding: `${theme.spacing.md}px ${theme.spacing.lg * 4}px ${theme.spacing.md}px ${theme.spacing.md}px`,
            fontFamily: 'monospace',
            fontSize: theme.typography.fontSizeSm,
            lineHeight: PROMPT_LINE_HEIGHT,
            color: theme.colors.textPrimary,
            whiteSpace: 'pre-wrap',
            overflowWrap: 'anywhere',
            ...(isExpanded
              ? {}
              : {
                  maxHeight: `calc(${PROMPT_LINE_HEIGHT * COLLAPSED_PROMPT_LINES}em + ${theme.spacing.md}px)`,
                  overflow: 'hidden',
                  maskImage: 'linear-gradient(to bottom, black 60%, transparent)',
                }),
          }}
        >
          {prompt}
        </pre>
        <div css={{ display: 'flex', justifyContent: 'center', borderTop: `1px solid ${theme.colors.border}` }}>
          <Button
            componentId="mlflow.experiment-issues.detail.proposed-fix.toggle"
            type="tertiary"
            size="small"
            endIcon={isExpanded ? <ChevronUpIcon /> : <ChevronDownIcon />}
            aria-expanded={isExpanded}
            aria-controls={promptId}
            onClick={() => setIsExpanded((expanded) => !expanded)}
          >
            {isExpanded ? (
              <FormattedMessage
                defaultMessage="Show less"
                description="Experiment issues tab > Issue detail > Fix this issue > Collapses the fix prompt"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Show more"
                description="Experiment issues tab > Issue detail > Fix this issue > Expands the whole fix prompt"
              />
            )}
          </Button>
        </div>
      </div>
    </section>
  );
};

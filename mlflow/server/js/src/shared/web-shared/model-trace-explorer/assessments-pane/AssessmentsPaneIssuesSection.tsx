import { Button, ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { IssueReferenceAssessment } from '../ModelTrace.types';
import { useCallback, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { isEmpty } from 'lodash';
import { GenAIMarkdownRenderer } from '../../genai-markdown-renderer/GenAIMarkdownRenderer';
import { useNavigate } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { RunPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { getIssue } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useGetIssueQuery';
import { SELECTED_ISSUE_ID_PARAM } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/RunViewIssuesTab';

const IssueItem = ({
  issue,
  onIssueClick,
}: {
  issue: IssueReferenceAssessment;
  onIssueClick: (issueId: string, issueName: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const issueName = issue.issue.issue_name || issue.assessment_name;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        marginBottom: theme.spacing.sm,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.sm + theme.spacing.xs,
        paddingTop: theme.spacing.sm,
        gap: theme.spacing.sm,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        {issue.rationale && (
          <Button
            componentId="shared.model-trace-explorer.toggle-issue-expanded"
            css={{ flexShrink: 0 }}
            size="small"
            icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
            onClick={() => setExpanded(!expanded)}
          />
        )}
        <span
          css={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            cursor: 'pointer',
            '&:hover': { textDecoration: 'underline' },
          }}
          title={issueName}
          onClick={(e: React.MouseEvent) => {
            e.stopPropagation();
            onIssueClick(issue.assessment_name, issueName);
          }}
        >
          <Typography.Text bold>{issueName}</Typography.Text>
        </span>
      </div>
      {expanded && issue.rationale && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
            paddingLeft: theme.spacing.lg / 2,
            marginLeft: theme.spacing.lg / 2,
            borderLeft: `1px solid ${theme.colors.border}`,
          }}
        >
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage defaultMessage="Rationale" description="Label for the rationale of an issue assessment" />
          </Typography.Text>
          <div css={{ '& > div:last-of-type': { marginBottom: 0 } }}>
            <GenAIMarkdownRenderer compact>{issue.rationale}</GenAIMarkdownRenderer>
          </div>
        </div>
      )}
    </div>
  );
};

export const AssessmentsPaneIssuesSection = ({ issues }: { issues: IssueReferenceAssessment[] }) => {
  const navigate = useNavigate();
  const sortedIssues = useMemo(
    () => issues.toSorted((left, right) => left.assessment_name.localeCompare(right.assessment_name)),
    [issues],
  );

  const { theme } = useDesignSystemTheme();

  const handleIssueClick = useCallback(
    async (issueId: string, _issueName: string) => {
      try {
        const issue = await getIssue(issueId);
        if (issue.source_run_id && issue.experiment_id) {
          const url = `${Routes.getIssueDetectionRunDetailsTabRoute(issue.experiment_id, issue.source_run_id, RunPageTabName.ISSUES)}?${SELECTED_ISSUE_ID_PARAM}=${encodeURIComponent(issueId)}`;
          navigate(url);
        }
      } catch (error) {
        console.error('Failed to fetch issue:', error);
      }
    },
    [navigate],
  );

  if (isEmpty(sortedIssues)) {
    return null;
  }

  return (
    <>
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          height: theme.spacing.lg,
          flexShrink: 0,
        }}
      >
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Issues"
            description="Label for the issues section in the assessments pane"
          />{' '}
          ({sortedIssues.length})
        </Typography.Text>
      </div>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        {sortedIssues.map((issue) => (
          <IssueItem issue={issue} key={issue.assessment_id} onIssueClick={handleIssueClick} />
        ))}
      </div>
    </>
  );
};

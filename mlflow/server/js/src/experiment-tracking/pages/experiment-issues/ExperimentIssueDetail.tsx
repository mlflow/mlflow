import { Alert, Button, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import Utils from '../../../common/utils/Utils';
import { GenAIMarkdownRenderer } from '../../../shared/web-shared/genai-markdown-renderer';
import { IssueTracesPanel } from '../../components/run-page/IssueTracesPanel';
import { STATUS_TAG_CONFIG, SEVERITY_TAG_CONFIG } from '../../components/run-page/IssueCard';
import type { Issue } from '../../components/run-page/hooks/useSearchIssuesQuery';
import Routes from '../../routes';
import { getIssueDescription, useTriageExperimentIssue } from './useExperimentIssues';

export const ExperimentIssueDetail = ({ issue, canEdit }: { issue: Issue; canEdit: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const triage = useTriageExperimentIssue(issue.experiment_id);
  const status = STATUS_TAG_CONFIG[issue.status];
  const severity = issue.severity ? SEVERITY_TAG_CONFIG[issue.severity] : undefined;
  return (
    <div css={{ flex: 1, minWidth: 0, overflow: 'auto', padding: theme.spacing.lg }}>
      <Typography.Title level={3}>{issue.name}</Typography.Title>
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.sm,
          alignItems: 'center',
          flexWrap: 'wrap',
          marginBottom: theme.spacing.md,
        }}
      >
        <Tag componentId="mlflow.experiment-issues.detail.status" color={status.color}>
          {intl.formatMessage(status.label)}
        </Tag>
        {severity && (
          <Tag componentId="mlflow.experiment-issues.detail.severity" color={severity.color}>
            {intl.formatMessage(severity.label)}
          </Tag>
        )}
        {issue.categories?.map((category) => (
          <Tag componentId="mlflow.experiment-issues.detail.category" key={category}>
            {category}
          </Tag>
        ))}
      </div>
      {canEdit && (
        <div css={{ display: 'flex', gap: theme.spacing.sm, marginBottom: theme.spacing.md }}>
          {issue.status !== 'resolved' && (
            <Button
              componentId="mlflow.experiment-issues.resolve"
              disabled={triage.isLoading}
              onClick={() => triage.mutate({ issueId: issue.issue_id, status: 'resolved' })}
            >
              <FormattedMessage defaultMessage="Resolve" description="Experiment Issues resolve action" />
            </Button>
          )}
          {issue.status !== 'rejected' && (
            <Button
              componentId="mlflow.experiment-issues.reject"
              disabled={triage.isLoading}
              onClick={() => triage.mutate({ issueId: issue.issue_id, status: 'rejected' })}
            >
              <FormattedMessage defaultMessage="Reject" description="Experiment Issues reject action" />
            </Button>
          )}
          {issue.status !== 'pending' && (
            <Button
              componentId="mlflow.experiment-issues.reopen"
              disabled={triage.isLoading}
              onClick={() => triage.mutate({ issueId: issue.issue_id, status: 'pending' })}
            >
              <FormattedMessage defaultMessage="Reopen" description="Experiment Issues reopen action" />
            </Button>
          )}
        </div>
      )}
      {triage.error instanceof Error && (
        <Alert
          componentId="mlflow.experiment-issues.triage-error"
          type="error"
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Unable to update issue"
              description="Experiment Issues triage error title"
            />
          }
          description={triage.error.message}
        />
      )}
      <GenAIMarkdownRenderer>{getIssueDescription(issue)}</GenAIMarkdownRenderer>
      <Typography.Paragraph color="secondary">
        <FormattedMessage
          defaultMessage="Last seen {time}"
          description="Experiment Issue last observation timestamp"
          values={{ time: Utils.formatTimestamp(issue.last_seen_timestamp ?? issue.created_timestamp) }}
        />
        {issue.source_run_id && (
          <>
            {' '}
            ·{' '}
            <Link
              componentId="mlflow.experiment-issues.source-run"
              to={Routes.getRunPageRoute(issue.experiment_id, issue.source_run_id)}
            >
              <FormattedMessage defaultMessage="Source run" description="Experiment Issue source run link" />
            </Link>
          </>
        )}
      </Typography.Paragraph>
      <Typography.Title level={4}>
        <FormattedMessage defaultMessage="Affected traces" description="Experiment Issue affected traces heading" />
      </Typography.Title>
      <div css={{ height: 500, display: 'flex', minWidth: 0 }}>
        <IssueTracesPanel issue={issue} experimentId={issue.experiment_id} />
      </div>
    </div>
  );
};

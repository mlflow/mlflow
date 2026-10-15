import {
  Alert,
  Button,
  ChevronDownIcon,
  DropdownMenu,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import Utils from '../../../common/utils/Utils';
import { GenAIMarkdownRenderer } from '../../../shared/web-shared/genai-markdown-renderer';
import { SEVERITY_TAG_CONFIG } from '../../components/run-page/IssueCard';
import type { Issue } from '../../components/run-page/hooks/useSearchIssuesQuery';
import Routes from '../../routes';
import { getIssueDescription, useTriageExperimentIssue } from './useExperimentIssues';
import { IssueCompletionIcon, IssueStatusMarker } from './ExperimentIssueRow';
import { IssueCategoryTag } from './IssueCategoryTag';
import { STATUS_LABELS } from './issueLabels';

export const ExperimentIssueDetail = ({ issue, canEdit }: { issue: Issue; canEdit: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const triage = useTriageExperimentIssue(issue.experiment_id);
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minWidth: 0,
        minHeight: 0,
        overflow: 'hidden',
        borderLeft: `1px solid ${theme.colors.border}`,
      }}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
          gap: theme.spacing.sm,
          padding: `${theme.spacing.md}px ${theme.spacing.lg}px`,
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.md }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flex: 1, minWidth: 0 }}>
            <IssueCompletionIcon status={issue.status} />
            <Typography.Title level={3} withoutMargins>
              {issue.name}
            </Typography.Title>
          </div>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="mlflow.experiment-issues.detail.status"
                size="small"
                endIcon={<ChevronDownIcon />}
                loading={triage.isLoading}
                disabled={!canEdit}
                css={{ flexShrink: 0 }}
              >
                <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <IssueStatusMarker status={issue.status} />
                  {intl.formatMessage(STATUS_LABELS[issue.status])}
                </span>
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="end">
              {issue.status === 'pending' ? (
                <>
                  <DropdownMenu.Item
                    componentId="mlflow.experiment-issues.resolve"
                    disabled={!canEdit || triage.isLoading}
                    onClick={() => triage.mutate({ issueId: issue.issue_id, status: 'resolved' })}
                  >
                    <FormattedMessage defaultMessage="Mark resolved" description="Experiment Issues resolve action" />
                  </DropdownMenu.Item>
                  <DropdownMenu.Item
                    componentId="mlflow.experiment-issues.reject"
                    disabled={!canEdit || triage.isLoading}
                    onClick={() => triage.mutate({ issueId: issue.issue_id, status: 'rejected' })}
                  >
                    <FormattedMessage defaultMessage="Not an issue" description="Experiment Issues reject action" />
                  </DropdownMenu.Item>
                </>
              ) : (
                <DropdownMenu.Item
                  componentId="mlflow.experiment-issues.reopen"
                  disabled={!canEdit || triage.isLoading}
                  onClick={() => triage.mutate({ issueId: issue.issue_id, status: 'pending' })}
                >
                  <FormattedMessage defaultMessage="Reopen" description="Experiment Issues reopen action" />
                </DropdownMenu.Item>
              )}
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
        <div css={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: theme.spacing.sm }}>
          {issue.severity && (
            <Typography.Text
              bold
              color={issue.severity === 'high' ? 'error' : issue.severity === 'medium' ? 'warning' : 'secondary'}
            >
              {intl.formatMessage(SEVERITY_TAG_CONFIG[issue.severity].label)}
            </Typography.Text>
          )}
          {issue.categories?.map((category) => (
            <IssueCategoryTag
              key={category}
              category={category}
              componentId="mlflow.experiment-issues.detail.category"
            />
          ))}
          <Typography.Text color="secondary">{Utils.timeSinceStr(issue.created_timestamp)}</Typography.Text>
        </div>
      </div>
      <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.lg,
            padding: `${theme.spacing.md + theme.spacing.xs}px ${theme.spacing.xl}px`,
            width: '100%',
            maxWidth: 960,
            marginInline: 'auto',
          }}
        >
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
          <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Title level={4} withoutMargins>
              <FormattedMessage defaultMessage="Description" description="Experiment Issue description heading" />
            </Typography.Title>
            <GenAIMarkdownRenderer compact>{getIssueDescription(issue)}</GenAIMarkdownRenderer>
            <Typography.Text color="secondary">
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
            </Typography.Text>
          </section>
        </div>
      </div>
    </div>
  );
};

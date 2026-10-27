import {
  Alert,
  Button,
  ChevronDownIcon,
  DropdownMenu,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { GenAIMarkdownRenderer } from '../../../shared/web-shared/genai-markdown-renderer';
import { FormattedMessage, useIntl } from 'react-intl';
import Utils from '../../../common/utils/Utils';
import type { Issue, IssueStatus } from '../../components/run-page/hooks/useSearchIssuesQuery';
import { IssueCompletionIcon, IssueStatusMarker } from './ExperimentIssueRow';
import { IssueCategoryTag } from './IssueCategoryTag';
import { SEVERITY_LABELS, STATUS_LABELS } from './issueLabels';
import { IssueDetailSectionTitle } from './IssueDetailSectionTitle';
import { IssueDetailColumn } from './IssueDetailColumn';
import { useTriageExperimentIssue } from './useExperimentIssues';
import { IssueEvidenceSection } from './IssueEvidenceSection';
import { IssueProposedFixSection } from './IssueProposedFixSection';

const IssueAge = ({ issue }: { issue: Issue }) => {
  const timeAgo = Utils.timeSinceStr(issue.created_timestamp);

  return <Typography.Text color="secondary">{timeAgo}</Typography.Text>;
};

export interface ExperimentIssueDetailProps {
  issue: Issue;
  onUpdateStatus: (status: IssueStatus) => void;
  isUpdatingStatus?: boolean;
}

/**
 * An issue's title, triage actions, and severity, categories and age: the part of the detail pane
 * that names the issue, pinned full width above the scrolling body.
 */
export const ExperimentIssueDetailHeader = ({
  issue,
  onUpdateStatus,
  isUpdatingStatus,
}: ExperimentIssueDetailProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const severityColor: 'error' | 'warning' | 'secondary' =
    issue.severity === 'high' ? 'error' : issue.severity === 'medium' ? 'warning' : 'secondary';

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.md }}>
        <div css={{ flex: 1, minWidth: 0 }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
            <IssueCompletionIcon status={issue.status} />
            <Typography.Title level={3} withoutMargins>
              {issue.name}
            </Typography.Title>
          </div>
        </div>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, flexShrink: 0 }}>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="mlflow.experiment-issues.detail.status"
                size="small"
                endIcon={<ChevronDownIcon />}
                loading={isUpdatingStatus}
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
                    componentId="mlflow.experiment-issues.detail.status.resolve"
                    disabled={isUpdatingStatus}
                    onClick={() => onUpdateStatus('resolved')}
                  >
                    <FormattedMessage defaultMessage="Mark resolved" description="Experiment issue resolve action" />
                  </DropdownMenu.Item>
                  <DropdownMenu.Item
                    componentId="mlflow.experiment-issues.detail.status.reject"
                    disabled={isUpdatingStatus}
                    onClick={() => onUpdateStatus('rejected')}
                  >
                    <FormattedMessage defaultMessage="Not an issue" description="Experiment issue reject action" />
                  </DropdownMenu.Item>
                </>
              ) : (
                <DropdownMenu.Item
                  componentId="mlflow.experiment-issues.detail.status.reopen"
                  disabled={isUpdatingStatus}
                  onClick={() => onUpdateStatus('pending')}
                >
                  <FormattedMessage defaultMessage="Reopen" description="Experiment issue reopen action" />
                </DropdownMenu.Item>
              )}
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </div>
      <div css={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: theme.spacing.sm }}>
        {issue.severity && (
          <Typography.Text bold color={severityColor}>
            {intl.formatMessage(SEVERITY_LABELS[issue.severity ?? 'not_an_issue'])}
          </Typography.Text>
        )}
        {issue.categories?.map((category) => (
          <IssueCategoryTag key={category} category={category} componentId="mlflow.experiment-issues.detail.category" />
        ))}
        <IssueAge issue={issue} />
      </div>
    </div>
  );
};

/**
 * The issue's description, rendered as Markdown. Traces and experiment editors shape it, so it uses the
 * trace drawer's renderer: an image from an untrusted host loads only when the reader asks, so a stored
 * image URL can't send data out from every reader's browser.
 */
export const ExperimentIssueDescription = ({ description }: { description: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <IssueDetailSectionTitle>
        <FormattedMessage
          defaultMessage="Description"
          description="Experiment issues tab > Issue detail > Description section heading"
        />
      </IssueDetailSectionTitle>
      <GenAIMarkdownRenderer compact>{description}</GenAIMarkdownRenderer>
    </div>
  );
};

export const ExperimentIssueDetail = ({
  issue,
  onViewTrace,
}: {
  issue: Issue;
  /** Opens one of the issue's traces, on the span its evidence blames where there is one. */
  onViewTrace: (linkableTraceId: string, spanId?: string) => void;
}) => {
  const triage = useTriageExperimentIssue(issue.experiment_id);
  return (
    <IssueDetailColumn
      issueKey={issue.issue_id}
      bodyMaxWidth={960}
      testId="experiment-issue-detail"
      header={
        <ExperimentIssueDetailHeader
          issue={issue}
          onUpdateStatus={(status) => triage.mutate({ issueId: issue.issue_id, status })}
          isUpdatingStatus={triage.isLoading}
        />
      }
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
      <ExperimentIssueDescription description={issue.description ?? ''} />
      <IssueEvidenceSection experimentId={issue.experiment_id} issueId={issue.issue_id} onViewTrace={onViewTrace} />
      <IssueProposedFixSection experimentId={issue.experiment_id} issue={issue} />
    </IssueDetailColumn>
  );
};

import { HoverCard, Overflow, Tag, useDesignSystemTheme } from '@databricks/design-system';
import { useNavigate } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { RunPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { getIssue } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useGetIssueQuery';
import { SELECTED_ISSUE_ID_PARAM } from '@mlflow/mlflow/src/experiment-tracking/constants';

import { NullCell } from './NullCell';
import { StackedComponents } from './StackedComponents';

export interface Issue {
  id: string;
  name: string;
}

const IssueTag = ({ issue }: { issue: Issue }) => {
  const navigate = useNavigate();

  const handleClick = async () => {
    try {
      const issueData = await getIssue(issue.id);
      if (issueData.source_run_id && issueData.experiment_id) {
        const baseUrl = Routes.getIssueDetectionRunDetailsTabRoute(
          issueData.experiment_id,
          issueData.source_run_id,
          RunPageTabName.ISSUES,
        );
        const params = new URLSearchParams({ [SELECTED_ISSUE_ID_PARAM]: issue.id });
        const url = `${baseUrl}?${params.toString()}`;
        navigate(url);
      }
    } catch (error) {
      // fail silently
    }
  };

  return (
    <Tag
      componentId="mlflow.genai-traces-table.issue-tag"
      color="coral"
      css={{ width: 'min-content', maxWidth: '100%', cursor: 'pointer' }}
      onClick={handleClick}
    >
      {issue.name}
    </Tag>
  );
};

const IssuesList = ({ issues, isComparing }: { issues: Issue[] | undefined; isComparing: boolean }) => {
  const { theme } = useDesignSystemTheme();

  if (!issues || issues.length === 0) {
    return <NullCell isComparing={isComparing} />;
  }

  const firstIssue = issues[0];
  const remainingIssues = issues.slice(1);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, minWidth: 0 }}>
      <Overflow>
        <IssueTag issue={firstIssue} />
      </Overflow>
      {remainingIssues.length > 0 && (
        <HoverCard
          trigger={
            <Tag
              componentId="mlflow.genai-traces-table.issue-tag-overflow-trigger"
              css={{
                cursor: 'default',
                width: 'fit-content',
              }}
            >
              +{remainingIssues.length}
            </Tag>
          }
          content={
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.xs,
                maxHeight: 200,
                overflowY: 'auto',
              }}
            >
              {remainingIssues.map((issue) => (
                <IssueTag key={issue.id} issue={issue} />
              ))}
            </div>
          }
        />
      )}
    </div>
  );
};

export const IssuesCell = ({
  issues,
  otherIssues,
  isComparing,
}: {
  issues: Issue[] | undefined;
  otherIssues: Issue[] | undefined;
  isComparing: boolean;
}) => {
  return (
    <StackedComponents
      first={<IssuesList issues={issues} isComparing={isComparing} />}
      second={isComparing && <IssuesList issues={otherIssues} isComparing={isComparing} />}
    />
  );
};

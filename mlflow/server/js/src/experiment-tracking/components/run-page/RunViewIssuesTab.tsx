import { TableSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { IssuesTabEmptyState } from './IssuesTabEmptyState';
import { IssueCard } from './IssueCard';
import { useSearchIssuesQuery } from './hooks/useSearchIssuesQuery';

export interface RunViewIssuesTabProps {
  runUuid: string;
  experimentId: string;
}

export const RunViewIssuesTab = ({ runUuid, experimentId }: RunViewIssuesTabProps) => {
  const { theme } = useDesignSystemTheme();
  const { issues, isLoading } = useSearchIssuesQuery({
    experimentId,
    sourceRunId: runUuid,
  });

  if (isLoading) {
    return (
      <div css={{ padding: theme.spacing.md }}>
        <TableSkeleton lines={5} />
      </div>
    );
  }

  if (issues.length === 0) {
    return (
      <div
        css={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.md,
        }}
      >
        <IssuesTabEmptyState />
      </div>
    );
  }

  return (
    <div
      css={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
        overflow: 'auto',
      }}
    >
      {issues.map((issue) => (
        <IssueCard key={issue.issue_id} issue={issue} />
      ))}
    </div>
  );
};

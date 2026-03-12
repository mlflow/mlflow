import { useMemo, useState } from 'react';
import { TableSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { IssuesTabEmptyState } from './IssuesTabEmptyState';
import { IssueCard } from './IssueCard';
import { IssueTracesPanel } from './IssueTracesPanel';
import { IssueStatusFilter, type IssueStatusFilterValue } from './IssueStatusFilter';
import { useSearchIssuesQuery, type Issue } from './hooks/useSearchIssuesQuery';

export interface RunViewIssuesTabProps {
  runUuid: string;
  experimentId: string;
}

export const RunViewIssuesTab = ({ runUuid, experimentId }: RunViewIssuesTabProps) => {
  const { theme } = useDesignSystemTheme();
  const [selectedIssue, setSelectedIssue] = useState<Issue | null>(null);
  const [statusFilter, setStatusFilter] = useState<IssueStatusFilterValue>('pending');
  const { issues, isLoading } = useSearchIssuesQuery({
    experimentId,
    sourceRunId: runUuid,
  });

  const filteredIssues = useMemo(() => {
    if (statusFilter === 'all') {
      return issues;
    }
    return issues.filter((issue) => issue.status === statusFilter);
  }, [issues, statusFilter]);

  const handleSelect = (issue: Issue) => {
    setSelectedIssue((prev) => (prev?.issue_id === issue.issue_id ? null : issue));
  };

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
        overflow: 'auto',
        minWidth: 320,
      }}
    >
      <IssueStatusFilter issues={issues} value={statusFilter} onChange={setStatusFilter} />
      <div
        css={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: selectedIssue ? 'minmax(280px, 1fr) 2fr' : '1fr',
          overflow: 'hidden',
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            padding: theme.spacing.md,
            overflow: 'auto',
          }}
        >
          {filteredIssues.map((issue) => (
            <IssueCard
              key={issue.issue_id}
              issue={issue}
              isSelected={selectedIssue?.issue_id === issue.issue_id}
              onSelect={() => handleSelect(issue)}
            />
          ))}
        </div>
        {selectedIssue && (
          <div
            css={{
              flex: 1,
              display: 'flex',
              borderLeft: `1px solid ${theme.colors.border}`,
              minHeight: 0,
              minWidth: 0,
              overflowY: 'auto',
            }}
          >
            <IssueTracesPanel issue={selectedIssue} experimentId={experimentId} />
          </div>
        )}
      </div>
    </div>
  );
};

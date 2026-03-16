import { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import { TableSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { IssuesTabEmptyState } from './IssuesTabEmptyState';
import { IssueCard } from './IssueCard';
import { IssueTracesPanel } from './IssueTracesPanel';
import { IssueStatusFilter, type IssueStatusFilterValue } from './IssueStatusFilter';
import { useSearchIssuesQuery, type Issue } from './hooks/useSearchIssuesQuery';
import { useSelectedIssueId } from './hooks/useSelectedIssueId';

export interface RunViewIssuesTabProps {
  runUuid: string;
  experimentId: string;
}

export const RunViewIssuesTab = ({ runUuid, experimentId }: RunViewIssuesTabProps) => {
  const { theme } = useDesignSystemTheme();
  const [statusFilter, setStatusFilter] = useState<IssueStatusFilterValue>('pending');
  const issueCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
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

  // Scroll to the selected issue card
  const scrollToSelectedIssue = useCallback((issueId: string) => {
    const cardElement = issueCardRefs.current[issueId];
    if (cardElement) {
      cardElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, []);

  const [selectedIssueId, setSelectedIssueId] = useSelectedIssueId({
    onSelect: scrollToSelectedIssue,
  });

  // Auto-select issue from URL parameter when issues load
  useEffect(() => {
    if (selectedIssueId && issues.length > 0) {
      const issue = issues.find((i) => i.issue_id === selectedIssueId);
      if (issue) {
        scrollToSelectedIssue(selectedIssueId);
      }
    }
  }, [selectedIssueId, issues, scrollToSelectedIssue]);

  const handleSelect = (issue: Issue) => {
    const isDeselecting = selectedIssueId === issue.issue_id;
    setSelectedIssueId(isDeselecting ? undefined : issue.issue_id);
  };

  const selectedIssue = useMemo(
    () => issues.find((i) => i.issue_id === selectedIssueId) || null,
    [issues, selectedIssueId],
  );

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
            <div key={issue.issue_id} ref={(el) => (issueCardRefs.current[issue.issue_id] = el)}>
              <IssueCard
                issue={issue}
                isSelected={selectedIssue?.issue_id === issue.issue_id}
                onSelect={() => handleSelect(issue)}
              />
            </div>
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

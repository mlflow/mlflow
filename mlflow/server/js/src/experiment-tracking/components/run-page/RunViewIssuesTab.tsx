import { useMemo, useState, useEffect, useRef } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { TableSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { IssuesTabEmptyState } from './IssuesTabEmptyState';
import { IssueCard } from './IssueCard';
import { IssueTracesPanel } from './IssueTracesPanel';
import { IssueStatusFilter, type IssueStatusFilterValue } from './IssueStatusFilter';
import { useSearchIssuesQuery, type Issue } from './hooks/useSearchIssuesQuery';
import { SELECTED_ISSUE_ID_PARAM } from '../../constants';

export interface RunViewIssuesTabProps {
  runUuid: string;
  experimentId: string;
}

export const RunViewIssuesTab = ({ runUuid, experimentId }: RunViewIssuesTabProps) => {
  const { theme } = useDesignSystemTheme();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedIssue, setSelectedIssue] = useState<Issue | null>(null);
  const [statusFilter, setStatusFilter] = useState<IssueStatusFilterValue>('pending');
  const issueCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const { issues, isLoading, refetch } = useSearchIssuesQuery({
    experimentId,
    sourceRunId: runUuid,
  });

  const filteredIssues = useMemo(() => {
    if (statusFilter === 'all') {
      return issues;
    }
    return issues.filter((issue) => issue.status === statusFilter);
  }, [issues, statusFilter]);

  // Auto-select issue from URL parameter when issues load or URL changes
  useEffect(() => {
    const issueIdFromUrl = searchParams.get(SELECTED_ISSUE_ID_PARAM);
    if (issueIdFromUrl && issues.length > 0 && selectedIssue?.issue_id !== issueIdFromUrl) {
      const issue = issues.find((i) => i.issue_id === issueIdFromUrl);
      if (issue) {
        setSelectedIssue(issue);
      }
    }
  }, [issues, searchParams, selectedIssue?.issue_id]);

  // Scroll to selected issue when it changes
  useEffect(() => {
    if (selectedIssue) {
      const cardElement = issueCardRefs.current[selectedIssue.issue_id];
      if (cardElement) {
        cardElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [selectedIssue]);

  const handleSelect = (issue: Issue) => {
    const isDeselecting = selectedIssue?.issue_id === issue.issue_id;
    setSelectedIssue(isDeselecting ? null : issue);

    // Update URL parameter
    if (isDeselecting) {
      searchParams.delete(SELECTED_ISSUE_ID_PARAM);
    } else {
      searchParams.set(SELECTED_ISSUE_ID_PARAM, issue.issue_id);
    }
    setSearchParams(searchParams, { replace: true });
  };

  const handleStatusUpdate = () => {
    refetch();
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
            <div key={issue.issue_id} ref={(el) => (issueCardRefs.current[issue.issue_id] = el)}>
              <IssueCard
                issue={issue}
                isSelected={selectedIssue?.issue_id === issue.issue_id}
                onSelect={() => handleSelect(issue)}
                onStatusUpdate={handleStatusUpdate}
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

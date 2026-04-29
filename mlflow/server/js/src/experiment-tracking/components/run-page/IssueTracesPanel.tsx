import { useDesignSystemTheme } from '@databricks/design-system';
import { useMemo } from 'react';
import type { TableFilter } from '@databricks/web-shared/genai-traces-table';
import { FilterOperator, ISSUE_ID_COLUMN_ID } from '@databricks/web-shared/genai-traces-table';
import { type Issue } from './hooks/useSearchIssuesQuery';
import { TracesV3Logs } from '../experiment-page/components/traces-v3/TracesV3Logs';

interface IssueTracesPanelProps {
  issue: Issue;
  experimentId: string;
}

export const IssueTracesPanel = ({ issue, experimentId }: IssueTracesPanelProps) => {
  const { theme } = useDesignSystemTheme();

  const issueFilter: TableFilter[] = useMemo(
    () => [
      {
        column: ISSUE_ID_COLUMN_ID,
        operator: FilterOperator.EQUALS,
        value: issue.issue_id,
      },
    ],
    [issue.issue_id],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        height: '100%',
        minHeight: 0,
        paddingLeft: theme.spacing.sm,
      }}
    >
      <TracesV3Logs
        experimentIds={[experimentId]}
        additionalFilters={issueFilter}
        disableActions
        columnStorageKeyPrefix="issue-traces"
      />
    </div>
  );
};

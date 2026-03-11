import { useMemo } from 'react';
import { SegmentedControlButton, SegmentedControlGroup, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { type Issue, type IssueStatus } from './hooks/useSearchIssuesQuery';

export type IssueStatusFilterValue = IssueStatus | 'all';

interface IssueStatusFilterProps {
  issues: Issue[];
  value: IssueStatusFilterValue;
  onChange: (value: IssueStatusFilterValue) => void;
}

export const IssueStatusFilter = ({ issues, value, onChange }: IssueStatusFilterProps) => {
  const { theme } = useDesignSystemTheme();

  const counts = useMemo(() => {
    const result = {
      all: issues.length,
      pending: 0,
      accepted: 0,
      rejected: 0,
      resolved: 0,
    };
    for (const issue of issues) {
      result[issue.status]++;
    }
    return result;
  }, [issues]);

  return (
    <div css={{ padding: `${theme.spacing.sm}px ${theme.spacing.md}px` }}>
      <SegmentedControlGroup name="mlflow.issues.status-filter" componentId="mlflow.issues.status-filter" value={value}>
        <SegmentedControlButton value="pending" onClick={() => onChange('pending')}>
          <FormattedMessage
            defaultMessage="Pending ({count})"
            description="Issue status filter > Pending button label"
            values={{ count: counts.pending }}
          />
        </SegmentedControlButton>
        <SegmentedControlButton value="accepted" onClick={() => onChange('accepted')}>
          <FormattedMessage
            defaultMessage="Accepted ({count})"
            description="Issue status filter > Accepted button label"
            values={{ count: counts.accepted }}
          />
        </SegmentedControlButton>
        <SegmentedControlButton value="rejected" onClick={() => onChange('rejected')}>
          <FormattedMessage
            defaultMessage="Rejected ({count})"
            description="Issue status filter > Rejected button label"
            values={{ count: counts.rejected }}
          />
        </SegmentedControlButton>
        <SegmentedControlButton value="resolved" onClick={() => onChange('resolved')}>
          <FormattedMessage
            defaultMessage="Resolved ({count})"
            description="Issue status filter > Resolved button label"
            values={{ count: counts.resolved }}
          />
        </SegmentedControlButton>
      </SegmentedControlGroup>
    </div>
  );
};

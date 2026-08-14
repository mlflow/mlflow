import { Button, ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { IssueReferenceAssessment } from '../ModelTrace.types';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { isEmpty } from 'lodash';
import { GenAIMarkdownRenderer } from '../../genai-markdown-renderer/GenAIMarkdownRenderer';

const IssueItem = ({ issue }: { issue: IssueReferenceAssessment }) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const issueName = issue.issue.issue_name || issue.assessment_name;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        marginBottom: theme.spacing.sm,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.sm + theme.spacing.xs,
        paddingTop: theme.spacing.sm,
        gap: theme.spacing.sm,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        {issue.rationale && (
          <Button
            componentId="shared.model-trace-explorer.toggle-issue-expanded"
            css={{ flexShrink: 0 }}
            size="small"
            icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
            onClick={() => setExpanded(!expanded)}
          />
        )}
        <Typography.Text
          bold
          css={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          title={issueName}
        >
          {issueName}
        </Typography.Text>
      </div>
      {expanded && issue.rationale && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
            paddingLeft: theme.spacing.lg / 2,
            marginLeft: theme.spacing.lg / 2,
            borderLeft: `1px solid ${theme.colors.border}`,
          }}
        >
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage defaultMessage="Rationale" description="Label for the rationale of an issue assessment" />
          </Typography.Text>
          <div css={{ '& > div:last-of-type': { marginBottom: 0 } }}>
            <GenAIMarkdownRenderer compact>{issue.rationale}</GenAIMarkdownRenderer>
          </div>
        </div>
      )}
    </div>
  );
};

export const AssessmentsPaneIssuesSection = ({
  issues,
  selectedIssueId,
}: {
  issues: IssueReferenceAssessment[];
  selectedIssueId?: string;
}) => {
  const sortedIssues = useMemo(
    () => issues.toSorted((left, right) => left.assessment_name.localeCompare(right.assessment_name)),
    [issues],
  );

  // Filter issues to only show the selected issue if one is selected
  const filteredIssues = useMemo(() => {
    if (!selectedIssueId) {
      return sortedIssues;
    }
    return sortedIssues.filter((issue) => {
      // issue_id is stored as assessment name
      return issue.assessment_name === selectedIssueId;
    });
  }, [sortedIssues, selectedIssueId]);

  const { theme } = useDesignSystemTheme();

  if (isEmpty(filteredIssues)) {
    return null;
  }

  return (
    <>
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          height: theme.spacing.lg,
          flexShrink: 0,
        }}
      >
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Issues"
            description="Label for the issues section in the assessments pane"
          />{' '}
          ({filteredIssues.length})
        </Typography.Text>
      </div>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        {filteredIssues.map((issue) => (
          <IssueItem issue={issue} key={issue.assessment_id} />
        ))}
      </div>
    </>
  );
};

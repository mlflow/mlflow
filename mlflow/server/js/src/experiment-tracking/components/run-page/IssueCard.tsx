import {
  Button,
  Card,
  CloseIcon,
  CopyIcon,
  CheckCircleIcon,
  InfoPopover,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { TagColors } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import Utils from '../../../common/utils/Utils';
import { type Issue, type IssueStatus } from './hooks/useSearchIssuesQuery';
import { useUpdateIssue } from './hooks/useUpdateIssue';

interface IssueCardProps {
  issue: Issue;
  isSelected: boolean;
  onSelect: () => void;
  onStatusUpdate?: () => void;
}

const STATUS_TAG_CONFIG: Record<IssueStatus, { color: TagColors; label: string }> = {
  pending: { color: 'lemon', label: 'Pending' },
  rejected: { color: 'coral', label: 'Rejected' },
  resolved: { color: 'purple', label: 'Resolved' },
};

export const IssueCard = ({ issue, isSelected, onSelect, onStatusUpdate }: IssueCardProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { updateIssue, isUpdating } = useUpdateIssue();

  const handleStatusChange = (newStatus: IssueStatus) => (e: React.MouseEvent) => {
    e.stopPropagation();
    updateIssue(
      { issueId: issue.issue_id, status: newStatus },
      {
        onSuccess: () => {
          onStatusUpdate?.();
        },
      },
    );
  };

  const statusConfig = STATUS_TAG_CONFIG[issue.status];

  return (
    <Card
      componentId="mlflow.issues.issue-card"
      css={{
        padding: theme.spacing.md,
        width: '100%',
        boxSizing: 'border-box',
        cursor: 'pointer',
        transition: 'box-shadow 0.2s ease, border-color 0.2s ease',
        border: isSelected ? `1px solid ${theme.colors.actionPrimaryBackgroundDefault}` : undefined,
      }}
      onClick={onSelect}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flexWrap: 'wrap' }}>
          <Typography.Title level={4} css={{ margin: 0, marginBottom: '0 !important' }}>
            {issue.name}
          </Typography.Title>
          <Tag componentId="mlflow.issues.status-tag" color={statusConfig.color} css={{ flexShrink: 0 }}>
            {statusConfig.label}
          </Tag>
          <InfoPopover iconTitle="Info" onClick={(e) => e.stopPropagation()}>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <div css={{ display: 'flex', alignItems: 'center', whiteSpace: 'nowrap' }}>
                <FormattedMessage defaultMessage="Issue ID" description="Label for issue ID in popover" />
                {': '}
                {issue.issue_id}{' '}
                <Typography.Text
                  size="md"
                  dangerouslySetAntdProps={{
                    copyable: {
                      text: issue.issue_id,
                      icon: <CopyIcon />,
                      tooltips: [
                        intl.formatMessage({
                          defaultMessage: 'Copy issue ID',
                          description: 'Tooltip to copy issue ID',
                        }),
                        intl.formatMessage({
                          defaultMessage: 'Issue ID copied',
                          description: 'Tooltip after issue ID was copied',
                        }),
                      ],
                    },
                  }}
                />
              </div>
            </div>
          </InfoPopover>
        </div>
        {issue.description && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text color="secondary" bold>
              <FormattedMessage defaultMessage="Description" description="Label for issue description section" />
            </Typography.Text>
            <Typography.Text
              css={
                isSelected
                  ? {}
                  : {
                      display: '-webkit-box',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      WebkitBoxOrient: 'vertical',
                      WebkitLineClamp: 2,
                    }
              }
            >
              {issue.description}
            </Typography.Text>
          </div>
        )}
        <Typography.Hint css={{ marginTop: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="Created: {date}"
            description="Issue creation date label"
            values={{ date: Utils.formatTimestamp(issue.created_timestamp, intl) }}
          />
        </Typography.Hint>
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.xs,
            marginTop: theme.spacing.sm,
            flexWrap: 'wrap',
            justifyContent: 'flex-end',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {issue.status === 'pending' && (
            <Button
              componentId="mlflow.issues.resolve-button"
              type="tertiary"
              size="small"
              icon={<CheckCircleIcon />}
              onClick={handleStatusChange('resolved')}
              loading={isUpdating}
            >
              <FormattedMessage defaultMessage="Resolve" description="Button to resolve an issue" />
            </Button>
          )}
          {issue.status === 'pending' && (
            <Button
              componentId="mlflow.issues.reject-button"
              type="tertiary"
              size="small"
              icon={<CloseIcon />}
              onClick={handleStatusChange('rejected')}
              loading={isUpdating}
            >
              <FormattedMessage defaultMessage="Reject" description="Button to reject an issue" />
            </Button>
          )}
          {(issue.status === 'resolved' || issue.status === 'rejected') && (
            <Button
              componentId="mlflow.issues.reopen-button"
              type="tertiary"
              size="small"
              onClick={handleStatusChange('pending')}
              loading={isUpdating}
            >
              <FormattedMessage defaultMessage="Reopen" description="Button to reopen an issue" />
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
};

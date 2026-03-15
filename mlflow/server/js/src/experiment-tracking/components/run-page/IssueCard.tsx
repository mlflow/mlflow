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
import { defineMessage, FormattedMessage, useIntl } from 'react-intl';
import type { MessageDescriptor } from 'react-intl';
import Utils from '../../../common/utils/Utils';
import { type Issue, type IssueStatus, type IssueSeverity } from './hooks/useSearchIssuesQuery';
import { useUpdateIssue } from './hooks/useUpdateIssue';
import { ISSUE_CATEGORY_DEFINITIONS } from '../experiment-page/components/traces-v3/IssueDetectionCategories';

interface IssueCardProps {
  issue: Issue;
  isSelected: boolean;
  onSelect: () => void;
}

const STATUS_TAG_CONFIG: Record<IssueStatus, { color: TagColors; label: MessageDescriptor }> = {
  pending: {
    color: 'lemon',
    label: defineMessage({ defaultMessage: 'Pending', description: 'Issue status tag label for pending issues' }),
  },
  rejected: {
    color: 'coral',
    label: defineMessage({ defaultMessage: 'Rejected', description: 'Issue status tag label for rejected issues' }),
  },
  resolved: {
    color: 'purple',
    label: defineMessage({ defaultMessage: 'Resolved', description: 'Issue status tag label for resolved issues' }),
  },
};

const SEVERITY_TAG_CONFIG: Record<IssueSeverity, { color: TagColors; label: MessageDescriptor }> = {
  not_an_issue: {
    color: 'charcoal',
    label: defineMessage({ defaultMessage: 'Not an issue', description: 'Issue severity tag label for not an issue' }),
  },
  low: {
    color: 'charcoal',
    label: defineMessage({ defaultMessage: 'Low', description: 'Issue severity tag label for low severity' }),
  },
  medium: {
    color: 'lemon',
    label: defineMessage({ defaultMessage: 'Medium', description: 'Issue severity tag label for medium severity' }),
  },
  high: {
    color: 'coral',
    label: defineMessage({ defaultMessage: 'High', description: 'Issue severity tag label for high severity' }),
  },
};

const parseCategoryId = (category: string): string => {
  return category.replace(/^\[|\]$/g, '');
};

const getCategoryLabel = (categoryId: string): React.ReactNode => {
  const definition = ISSUE_CATEGORY_DEFINITIONS.find((def) => def.id === categoryId);
  return definition?.title ?? categoryId;
};

export const IssueCard = ({ issue, isSelected, onSelect }: IssueCardProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { updateIssue, isUpdating } = useUpdateIssue();

  const handleStatusChange = (newStatus: IssueStatus) => (e: React.MouseEvent) => {
    e.stopPropagation();
    updateIssue({ issueId: issue.issue_id, status: newStatus });
  };

  const statusConfig = STATUS_TAG_CONFIG[issue.status];
  const severityConfig = issue.severity ? SEVERITY_TAG_CONFIG[issue.severity] : undefined;

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
        <div
          css={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: theme.spacing.sm }}
        >
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flexWrap: 'wrap', flex: 1 }}>
            {severityConfig && (
              <Tag componentId="mlflow.issues.severity-tag" color={severityConfig.color} css={{ flexShrink: 0 }}>
                {intl.formatMessage(severityConfig.label)}
              </Tag>
            )}
            <Typography.Title level={4} css={{ margin: 0, marginBottom: '0 !important' }}>
              {issue.name}
            </Typography.Title>
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
          <Tag componentId="mlflow.issues.status-tag" color={statusConfig.color} css={{ flexShrink: 0 }}>
            {intl.formatMessage(statusConfig.label)}
          </Tag>
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
            alignItems: 'center',
            justifyContent: 'space-between',
            marginTop: theme.spacing.sm,
            flexWrap: 'wrap',
            gap: theme.spacing.sm,
          }}
        >
          {issue.categories && issue.categories.length > 0 && (
            <div css={{ display: 'flex', gap: theme.spacing.xs, flexWrap: 'wrap' }}>
              {issue.categories.map((category) => {
                const categoryId = parseCategoryId(category);
                return (
                  <Tag key={category} componentId="mlflow.issues.category-tag" color="turquoise">
                    {getCategoryLabel(categoryId)}
                  </Tag>
                );
              })}
            </div>
          )}
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.xs,
              flexWrap: 'wrap',
              marginLeft: 'auto',
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
                componentId="mlflow.issues.move-to-pending-button"
                type="tertiary"
                size="small"
                onClick={handleStatusChange('pending')}
                loading={isUpdating}
              >
                <FormattedMessage
                  defaultMessage="Move to pending"
                  description="Button to move an issue back to pending status"
                />
              </Button>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};

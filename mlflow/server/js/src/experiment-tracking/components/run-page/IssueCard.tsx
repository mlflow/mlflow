import { Card, CopyIcon, InfoPopover, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import Utils from '../../../common/utils/Utils';
import { type Issue } from './hooks/useSearchIssuesQuery';

export const IssueCard = ({ issue }: { issue: Issue }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <Card
      componentId="mlflow.issues.issue-card"
      css={{
        padding: theme.spacing.md,
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: '1fr auto',
          gap: theme.spacing.xs,
          alignItems: 'flex-start',
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Typography.Title level={4} css={{ margin: 0, marginBottom: '0 !important' }}>
              {issue.name}
            </Typography.Title>
            <InfoPopover iconTitle="Info">
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
            <Typography.Text
              color="secondary"
              css={{
                display: '-webkit-box',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                WebkitBoxOrient: 'vertical',
                WebkitLineClamp: 2,
              }}
              title={issue.description}
            >
              {issue.description}
            </Typography.Text>
          )}
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="Created: {date}"
              description="Issue creation date label"
              values={{ date: Utils.formatTimestamp(issue.created_timestamp, intl) }}
            />
          </Typography.Hint>
        </div>
      </div>
    </Card>
  );
};

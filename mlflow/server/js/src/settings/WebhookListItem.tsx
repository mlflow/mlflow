import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import type { Webhook, WebhookEvent } from './webhooksApi';
import { eventKey, eventLabels } from './webhookConstants';

interface WebhookListItemProps {
  webhook: Webhook;
  isLast: boolean;
  testingId: string | null;
  deletingId: string | null;
  onTest: (webhook: Webhook) => void;
  onEdit: (webhook: Webhook) => void;
  onDelete: (webhook: Webhook) => void;
}

const formatEventLabel = (intl: ReturnType<typeof useIntl>, entity: string, action: string) => {
  const key = eventKey(entity, action);
  const descriptor = eventLabels[key as keyof typeof eventLabels];
  return descriptor ? intl.formatMessage(descriptor) : key;
};

const formatEvents = (intl: ReturnType<typeof useIntl>, events: WebhookEvent[]) =>
  events.map((e) => formatEventLabel(intl, e.entity, e.action)).join(', ');

const WebhookListItem = ({
  webhook,
  isLast,
  testingId,
  deletingId,
  onTest,
  onEdit,
  onDelete,
}: WebhookListItemProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div
      css={{
        padding: theme.spacing.md,
        borderBottom: isLast ? 'none' : `1px solid ${theme.colors.border}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        gap: theme.spacing.md,
      }}
    >
      <div css={{ flex: 1, minWidth: 0 }}>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            flexWrap: 'wrap',
          }}
        >
          <Typography.Text bold>{webhook.name}</Typography.Text>
          <span
            css={{
              padding: `2px ${theme.spacing.xs}px`,
              borderRadius: theme.legacyBorders.borderRadiusMd,
              fontSize: theme.typography.fontSizeSm,
              backgroundColor: webhook.status === 'ACTIVE' ? theme.colors.green100 : theme.colors.grey200,
              color: webhook.status === 'ACTIVE' ? theme.colors.green700 : theme.colors.textSecondary,
            }}
          >
            {webhook.status === 'ACTIVE'
              ? intl.formatMessage({ defaultMessage: 'Active', description: 'Webhook active status' })
              : intl.formatMessage({ defaultMessage: 'Disabled', description: 'Webhook disabled status' })}
          </span>
        </div>
        <Typography.Text size="sm" color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
          {webhook.url}
        </Typography.Text>
        <Typography.Text size="sm" color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
          {formatEvents(intl, webhook.events)}
        </Typography.Text>
      </div>
      <div css={{ display: 'flex', gap: theme.spacing.sm, flexShrink: 0 }}>
        <Button
          componentId="mlflow.settings.webhooks.test-button"
          size="small"
          onClick={() => onTest(webhook)}
          loading={testingId === webhook.webhook_id}
          disabled={testingId !== null || deletingId === webhook.webhook_id}
        >
          <FormattedMessage defaultMessage="Test" description="Test webhook button" />
        </Button>
        <Button
          componentId="mlflow.settings.webhooks.edit-button"
          size="small"
          onClick={() => onEdit(webhook)}
          disabled={deletingId === webhook.webhook_id}
        >
          <FormattedMessage defaultMessage="Edit" description="Edit webhook button" />
        </Button>
        <Button
          componentId="mlflow.settings.webhooks.delete-button"
          size="small"
          danger
          onClick={() => onDelete(webhook)}
          loading={deletingId === webhook.webhook_id}
          disabled={testingId !== null}
        >
          <FormattedMessage defaultMessage="Delete" description="Delete webhook button" />
        </Button>
      </div>
    </div>
  );
};

export default WebhookListItem;

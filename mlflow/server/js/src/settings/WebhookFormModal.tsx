import { useCallback, useState } from 'react';
import { Alert, Checkbox, Input, Modal, Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { WebhooksApi } from './webhooksApi';
import type { Webhook, WebhookEvent } from './webhooksApi';
import {
  VALID_EVENTS,
  WEBHOOK_NAME_REGEX,
  EMPTY_FORM,
  eventKey,
  eventLabels,
  type WebhookFormState,
} from './webhookConstants';

interface WebhookFormModalProps {
  visible: boolean;
  editingWebhook: Webhook | null;
  onClose: () => void;
  onSaved: () => void;
}

const WebhookFormModal = ({ visible, editingWebhook, onClose, onSaved }: WebhookFormModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [form, setForm] = useState<WebhookFormState>(() => {
    if (editingWebhook) {
      return {
        name: editingWebhook.name,
        url: editingWebhook.url,
        description: editingWebhook.description ?? '',
        secret: '',
        status: editingWebhook.status === 'ACTIVE',
        events: new Set(editingWebhook.events.map((e) => eventKey(e.entity, e.action))),
      };
    }
    return EMPTY_FORM;
  });
  const [formError, setFormError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const formatEventLabel = (entity: string, action: string) => {
    const key = eventKey(entity, action);
    const descriptor = eventLabels[key as keyof typeof eventLabels];
    return descriptor ? intl.formatMessage(descriptor) : key;
  };

  const handleSave = useCallback(async () => {
    if (!form.name.trim()) {
      setFormError(intl.formatMessage({ defaultMessage: 'Name is required' }));
      return;
    }
    const trimmedName = form.name.trim();
    if (trimmedName.length > 63 || !WEBHOOK_NAME_REGEX.test(trimmedName)) {
      setFormError(
        intl.formatMessage({
          defaultMessage:
            'Name must start and end with a letter or digit, be less than 63 characters long, and contain only letters, digits, dots (.), underscores (_), and hyphens (-).',
          description: 'Webhook name validation error',
        }),
      );
      return;
    }
    if (!form.url.trim()) {
      setFormError(intl.formatMessage({ defaultMessage: 'URL is required' }));
      return;
    }
    if (form.events.size === 0) {
      setFormError(intl.formatMessage({ defaultMessage: 'At least one event must be selected' }));
      return;
    }

    const events: WebhookEvent[] = Array.from(form.events).map((key) => {
      const [entity, action] = key.split('.', 2);
      return { entity, action };
    });

    setIsSaving(true);
    setFormError(null);

    const payload = {
      name: form.name.trim(),
      url: form.url.trim(),
      events,
      description: form.description.trim() || undefined,
      secret: form.secret.trim() || undefined,
      status: form.status ? ('ACTIVE' as const) : ('DISABLED' as const),
    };

    try {
      if (editingWebhook) {
        await WebhooksApi.updateWebhook(editingWebhook.webhook_id, payload);
      } else {
        await WebhooksApi.createWebhook(payload);
      }
      onSaved();
    } catch (e: any) {
      setFormError(e?.message ?? intl.formatMessage({ defaultMessage: 'Failed to save webhook' }));
    } finally {
      setIsSaving(false);
    }
  }, [form, editingWebhook, onSaved, intl]);

  return (
    <Modal
      componentId="mlflow.settings.webhooks.form-modal"
      title={
        editingWebhook
          ? intl.formatMessage({ defaultMessage: 'Edit webhook', description: 'Edit webhook modal title' })
          : intl.formatMessage({ defaultMessage: 'Create webhook', description: 'Create webhook modal title' })
      }
      visible={visible}
      onCancel={onClose}
      onOk={handleSave}
      okText={
        editingWebhook
          ? intl.formatMessage({ defaultMessage: 'Save', description: 'Save webhook button' })
          : intl.formatMessage({ defaultMessage: 'Create', description: 'Create webhook confirm button' })
      }
      cancelText={intl.formatMessage({ defaultMessage: 'Cancel', description: 'Cancel webhook form button' })}
      confirmLoading={isSaving}
      size="wide"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {formError && (
          <Alert componentId="mlflow.settings.webhooks.form-error-alert" type="error" message={formError} />
        )}

        <div>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Name" description="Webhook name field label" />{' '}
            <Typography.Text color="error">*</Typography.Text>
          </Typography.Text>
          <Input
            componentId="mlflow.settings.webhooks.name-input"
            value={form.name}
            onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
            placeholder={intl.formatMessage({ defaultMessage: 'My webhook', description: 'Webhook name placeholder' })}
            css={{ marginTop: theme.spacing.xs }}
          />
        </div>

        <div>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="URL" description="Webhook URL field label" />{' '}
            <Typography.Text color="error">*</Typography.Text>
          </Typography.Text>
          <Input
            componentId="mlflow.settings.webhooks.url-input"
            value={form.url}
            onChange={(e) => setForm((prev) => ({ ...prev, url: e.target.value }))}
            placeholder={intl.formatMessage({
              defaultMessage: 'https://example.com/webhook',
              description: 'Webhook URL placeholder',
            })}
            css={{ marginTop: theme.spacing.xs }}
          />
        </div>

        <div>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Description" description="Webhook description field label" />
          </Typography.Text>
          <Input.TextArea
            componentId="mlflow.settings.webhooks.description-input"
            value={form.description}
            onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
            placeholder={intl.formatMessage({
              defaultMessage: 'Optional description',
              description: 'Webhook description placeholder',
            })}
            rows={2}
            css={{ marginTop: theme.spacing.xs }}
          />
        </div>

        <div>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Secret" description="Webhook secret field label" />
          </Typography.Text>
          <Typography.Text size="sm" color="secondary" css={{ display: 'block' }}>
            <FormattedMessage
              defaultMessage="Used for HMAC signature verification of incoming webhook requests."
              description="Webhook secret field description"
            />
          </Typography.Text>
          <Input
            componentId="mlflow.settings.webhooks.secret-input"
            type="password"
            value={form.secret}
            onChange={(e) => setForm((prev) => ({ ...prev, secret: e.target.value }))}
            placeholder={
              editingWebhook
                ? intl.formatMessage({
                    defaultMessage: 'Leave blank to keep existing secret',
                    description: 'Webhook secret placeholder when editing',
                  })
                : intl.formatMessage({
                    defaultMessage: 'Optional secret key',
                    description: 'Webhook secret placeholder when creating',
                  })
            }
            css={{ marginTop: theme.spacing.xs }}
          />
        </div>

        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Active" description="Webhook status field label" />
          </Typography.Text>
          <Switch
            componentId="mlflow.settings.webhooks.status-switch"
            checked={form.status}
            onChange={(checked) => setForm((prev) => ({ ...prev, status: checked }))}
            label={
              form.status
                ? intl.formatMessage({ defaultMessage: 'Active', description: 'Webhook active status label' })
                : intl.formatMessage({ defaultMessage: 'Disabled', description: 'Webhook disabled status label' })
            }
            activeLabel={intl.formatMessage({ defaultMessage: 'Active', description: 'Webhook active label' })}
            inactiveLabel={intl.formatMessage({ defaultMessage: 'Disabled', description: 'Webhook disabled label' })}
          />
        </div>

        <div>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Events" description="Webhook events field label" />{' '}
            <Typography.Text color="error">*</Typography.Text>
          </Typography.Text>
          <Typography.Text size="sm" color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Select the events that will trigger this webhook."
              description="Webhook events field description"
            />
          </Typography.Text>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            {VALID_EVENTS.map((event) => {
              const key = eventKey(event.entity, event.action);
              return (
                <Checkbox
                  key={key}
                  componentId={`mlflow.settings.webhooks.event-${key}`}
                  isChecked={form.events.has(key)}
                  onChange={(checked) => {
                    setForm((prev) => {
                      const next = new Set(prev.events);
                      if (checked) {
                        next.add(key);
                      } else {
                        next.delete(key);
                      }
                      return { ...prev, events: next };
                    });
                  }}
                >
                  {formatEventLabel(event.entity, event.action)}
                </Checkbox>
              );
            })}
          </div>
        </div>
      </div>
    </Modal>
  );
};

export default WebhookFormModal;

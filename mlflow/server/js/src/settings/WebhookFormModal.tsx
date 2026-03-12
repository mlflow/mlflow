import { useState } from 'react';
import { useForm, Controller, FormProvider } from 'react-hook-form';
import {
  Alert,
  Checkbox,
  FormUI,
  Modal,
  RHFControlledComponents,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { WebhooksApi } from './webhooksApi';
import type { Webhook, WebhookEvent } from './webhooksApi';
import { VALID_EVENTS, WEBHOOK_NAME_REGEX, eventKey, eventLabels } from './webhookConstants';

interface WebhookFormData {
  name: string;
  url: string;
  description: string;
  secret: string;
  status: boolean;
  events: string[];
}

interface WebhookFormModalProps {
  visible: boolean;
  editingWebhook: Webhook | null;
  onClose: () => void;
  onSaved: () => void;
  /** If set, only show events matching this entity */
  eventFilter?: string;
}

const WebhookFormModal = ({ visible, editingWebhook, onClose, onSaved, eventFilter }: WebhookFormModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const form = useForm<WebhookFormData>({
    defaultValues: {
      name: editingWebhook?.name ?? '',
      url: editingWebhook?.url ?? '',
      description: editingWebhook?.description ?? '',
      secret: '',
      status: editingWebhook ? editingWebhook.status === 'ACTIVE' : true,
      events: editingWebhook?.events.map((e) => eventKey(e.entity, e.action)) ?? [],
    },
  });

  const displayedEvents = eventFilter ? VALID_EVENTS.filter((e) => e.entity === eventFilter) : VALID_EVENTS;

  const formatEventLabel = (entity: string, action: string) => {
    const key = eventKey(entity, action);
    const descriptor = eventLabels[key as keyof typeof eventLabels];
    return descriptor ? intl.formatMessage(descriptor) : key;
  };

  const handleSubmit = async (values: WebhookFormData) => {
    const events: WebhookEvent[] = values.events.map((key) => {
      const [entity, action] = key.split('.', 2);
      return { entity, action };
    });

    setIsSaving(true);
    setSubmitError(null);

    const payload = {
      name: values.name.trim(),
      url: values.url.trim(),
      events,
      description: values.description.trim() || undefined,
      secret: values.secret.trim() || undefined,
      status: values.status ? ('ACTIVE' as const) : ('DISABLED' as const),
    };

    try {
      if (editingWebhook) {
        await WebhooksApi.updateWebhook(editingWebhook.webhook_id, payload);
      } else {
        await WebhooksApi.createWebhook(payload);
      }
      onSaved();
    } catch (e: any) {
      setSubmitError(e?.message ?? intl.formatMessage({ defaultMessage: 'Failed to save webhook' }));
    } finally {
      setIsSaving(false);
    }
  };

  const nameValidationMessage = intl.formatMessage({
    defaultMessage:
      'Name must start and end with a letter or digit, be less than 63 characters long, and contain only letters, digits, dots (.), underscores (_), and hyphens (-).',
    description: 'Webhook name validation error',
  });

  return (
    <FormProvider {...form}>
      <Modal
        componentId="mlflow.settings.webhooks.form-modal"
        title={
          editingWebhook
            ? intl.formatMessage({ defaultMessage: 'Edit webhook', description: 'Edit webhook modal title' })
            : intl.formatMessage({ defaultMessage: 'Create webhook', description: 'Create webhook modal title' })
        }
        visible={visible}
        onCancel={onClose}
        onOk={form.handleSubmit(handleSubmit)}
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
          {submitError && (
            <Alert componentId="mlflow.settings.webhooks.form-error-alert" type="error" message={submitError} />
          )}

          <div>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.name-input">
              <FormattedMessage defaultMessage="Name" description="Webhook name field label" /> *
            </FormUI.Label>
            <RHFControlledComponents.Input
              name="name"
              control={form.control}
              id="mlflow.settings.webhooks.name-input"
              componentId="mlflow.settings.webhooks.name-input"
              placeholder={intl.formatMessage({
                defaultMessage: 'My webhook',
                description: 'Webhook name placeholder',
              })}
              rules={{
                required: intl.formatMessage({ defaultMessage: 'Name is required' }),
                validate: (value) => {
                  const trimmed = String(value).trim();
                  if (trimmed.length > 63 || !WEBHOOK_NAME_REGEX.test(trimmed)) {
                    return nameValidationMessage;
                  }
                  return true;
                },
              }}
              validationState={form.formState.errors.name ? 'error' : undefined}
            />
            {form.formState.errors.name && <FormUI.Message type="error" message={form.formState.errors.name.message} />}
          </div>

          <div>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.url-input">
              <FormattedMessage defaultMessage="URL" description="Webhook URL field label" /> *
            </FormUI.Label>
            <RHFControlledComponents.Input
              name="url"
              control={form.control}
              id="mlflow.settings.webhooks.url-input"
              componentId="mlflow.settings.webhooks.url-input"
              placeholder={intl.formatMessage({
                defaultMessage: 'https://example.com/webhook',
                description: 'Webhook URL placeholder',
              })}
              rules={{
                required: intl.formatMessage({ defaultMessage: 'URL is required' }),
              }}
              validationState={form.formState.errors.url ? 'error' : undefined}
            />
            {form.formState.errors.url && <FormUI.Message type="error" message={form.formState.errors.url.message} />}
          </div>

          <div>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.description-input">
              <FormattedMessage defaultMessage="Description" description="Webhook description field label" />
            </FormUI.Label>
            <RHFControlledComponents.TextArea
              name="description"
              control={form.control}
              id="mlflow.settings.webhooks.description-input"
              componentId="mlflow.settings.webhooks.description-input"
              placeholder={intl.formatMessage({
                defaultMessage: 'Optional description',
                description: 'Webhook description placeholder',
              })}
              rows={2}
            />
          </div>

          <div>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.secret-input">
              <FormattedMessage defaultMessage="Secret" description="Webhook secret field label" />
            </FormUI.Label>
            <FormUI.Hint>
              <FormattedMessage
                defaultMessage="Used for HMAC signature verification of incoming webhook requests."
                description="Webhook secret field description"
              />
            </FormUI.Hint>
            <RHFControlledComponents.Password
              name="secret"
              control={form.control}
              id="mlflow.settings.webhooks.secret-input"
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
            />
          </div>

          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <FormUI.Label htmlFor="mlflow.settings.webhooks.status-switch">
              <FormattedMessage defaultMessage="Active" description="Webhook status field label" />
            </FormUI.Label>
            <RHFControlledComponents.Switch
              name="status"
              control={form.control}
              id="mlflow.settings.webhooks.status-switch"
              componentId="mlflow.settings.webhooks.status-switch"
              activeLabel={intl.formatMessage({ defaultMessage: 'Active', description: 'Webhook active label' })}
              inactiveLabel={intl.formatMessage({ defaultMessage: 'Disabled', description: 'Webhook disabled label' })}
            />
          </div>

          <div>
            <FormUI.Label>
              <FormattedMessage defaultMessage="Events" description="Webhook events field label" /> *
            </FormUI.Label>
            <FormUI.Hint>
              <FormattedMessage
                defaultMessage="Select the events that will trigger this webhook."
                description="Webhook events field description"
              />
            </FormUI.Hint>
            <Controller
              name="events"
              control={form.control}
              rules={{
                validate: (value: string[]) =>
                  value.length > 0 || intl.formatMessage({ defaultMessage: 'At least one event must be selected' }),
              }}
              render={({ field }) => (
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  {displayedEvents.map((event) => {
                    const key = eventKey(event.entity, event.action);
                    return (
                      <Checkbox
                        key={key}
                        componentId={`mlflow.settings.webhooks.event-${key}`}
                        isChecked={field.value.includes(key)}
                        onChange={(checked) => {
                          const next = checked ? [...field.value, key] : field.value.filter((k: string) => k !== key);
                          field.onChange(next);
                        }}
                      >
                        {formatEventLabel(event.entity, event.action)}
                      </Checkbox>
                    );
                  })}
                </div>
              )}
            />
            {form.formState.errors.events && (
              <FormUI.Message type="error" message={form.formState.errors.events.message} />
            )}
          </div>
        </div>
      </Modal>
    </FormProvider>
  );
};

export default WebhookFormModal;

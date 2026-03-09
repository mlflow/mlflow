import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Button,
  Checkbox,
  Input,
  Modal,
  Spinner,
  Switch,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { defineMessages, FormattedMessage, useIntl } from '@databricks/i18n';
import { WebhooksApi } from './webhooksApi';
import type { Webhook, WebhookEvent } from './webhooksApi';

interface WebhookFormState {
  name: string;
  url: string;
  description: string;
  secret: string;
  status: boolean; // true = ACTIVE
  events: Set<string>;
}

const eventLabels = defineMessages({
  'REGISTERED_MODEL.CREATED': { defaultMessage: 'Registered model created', description: 'Webhook event label' },
  'MODEL_VERSION.CREATED': { defaultMessage: 'Model version created', description: 'Webhook event label' },
  'MODEL_VERSION_TAG.SET': { defaultMessage: 'Model version tag set', description: 'Webhook event label' },
  'MODEL_VERSION_TAG.DELETED': { defaultMessage: 'Model version tag deleted', description: 'Webhook event label' },
  'MODEL_VERSION_ALIAS.CREATED': { defaultMessage: 'Model version alias created', description: 'Webhook event label' },
  'MODEL_VERSION_ALIAS.DELETED': { defaultMessage: 'Model version alias deleted', description: 'Webhook event label' },
  'PROMPT.CREATED': { defaultMessage: 'Prompt created', description: 'Webhook event label' },
  'PROMPT_VERSION.CREATED': { defaultMessage: 'Prompt version created', description: 'Webhook event label' },
  'PROMPT_TAG.SET': { defaultMessage: 'Prompt tag set', description: 'Webhook event label' },
  'PROMPT_TAG.DELETED': { defaultMessage: 'Prompt tag deleted', description: 'Webhook event label' },
  'PROMPT_VERSION_TAG.SET': { defaultMessage: 'Prompt version tag set', description: 'Webhook event label' },
  'PROMPT_VERSION_TAG.DELETED': { defaultMessage: 'Prompt version tag deleted', description: 'Webhook event label' },
  'PROMPT_ALIAS.CREATED': { defaultMessage: 'Prompt alias created', description: 'Webhook event label' },
  'PROMPT_ALIAS.DELETED': { defaultMessage: 'Prompt alias deleted', description: 'Webhook event label' },
  'BUDGET_POLICY.EXCEEDED': { defaultMessage: 'Budget policy exceeded', description: 'Webhook event label' },
});

const VALID_EVENTS: { entity: string; action: string }[] = [
  { entity: 'REGISTERED_MODEL', action: 'CREATED' },
  { entity: 'MODEL_VERSION', action: 'CREATED' },
  { entity: 'MODEL_VERSION_TAG', action: 'SET' },
  { entity: 'MODEL_VERSION_TAG', action: 'DELETED' },
  { entity: 'MODEL_VERSION_ALIAS', action: 'CREATED' },
  { entity: 'MODEL_VERSION_ALIAS', action: 'DELETED' },
  { entity: 'PROMPT', action: 'CREATED' },
  { entity: 'PROMPT_VERSION', action: 'CREATED' },
  { entity: 'PROMPT_TAG', action: 'SET' },
  { entity: 'PROMPT_TAG', action: 'DELETED' },
  { entity: 'PROMPT_VERSION_TAG', action: 'SET' },
  { entity: 'PROMPT_VERSION_TAG', action: 'DELETED' },
  { entity: 'PROMPT_ALIAS', action: 'CREATED' },
  { entity: 'PROMPT_ALIAS', action: 'DELETED' },
  { entity: 'BUDGET_POLICY', action: 'EXCEEDED' },
];

const eventKey = (entity: string, action: string) => `${entity}.${action}`;

const EMPTY_FORM: WebhookFormState = {
  name: '',
  url: '',
  description: '',
  secret: '',
  status: true,
  events: new Set(),
};

const WebhooksSettings = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [webhooks, setWebhooks] = useState<Webhook[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingWebhook, setEditingWebhook] = useState<Webhook | null>(null);
  const [form, setForm] = useState<WebhookFormState>(EMPTY_FORM);
  const [formError, setFormError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [webhookToDelete, setWebhookToDelete] = useState<Webhook | null>(null);

  const [testingId, setTestingId] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<{
    webhookId: string;
    success: boolean;
    message: string;
  } | null>(null);

  const fetchWebhooks = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await WebhooksApi.listWebhooks();
      setWebhooks(response?.webhooks ?? []);
    } catch (e: any) {
      setError(e?.message ?? intl.formatMessage({ defaultMessage: 'Failed to load webhooks' }));
    } finally {
      setLoading(false);
    }
  }, [intl]);

  useEffect(() => {
    fetchWebhooks();
  }, [fetchWebhooks]);

  const openCreateModal = useCallback(() => {
    setEditingWebhook(null);
    setForm(EMPTY_FORM);
    setFormError(null);
    setIsModalOpen(true);
  }, []);

  const openEditModal = useCallback((webhook: Webhook) => {
    setEditingWebhook(webhook);
    setForm({
      name: webhook.name,
      url: webhook.url,
      description: webhook.description ?? '',
      secret: '',
      status: webhook.status === 'ACTIVE',
      events: new Set(webhook.events.map((e) => eventKey(e.entity, e.action))),
    });
    setFormError(null);
    setIsModalOpen(true);
  }, []);

  const closeModal = useCallback(() => {
    setIsModalOpen(false);
    setEditingWebhook(null);
    setForm(EMPTY_FORM);
    setFormError(null);
  }, []);

  const handleSave = useCallback(async () => {
    if (!form.name.trim()) {
      setFormError(intl.formatMessage({ defaultMessage: 'Name is required' }));
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
      closeModal();
      await fetchWebhooks();
    } catch (e: any) {
      setFormError(e?.message ?? intl.formatMessage({ defaultMessage: 'Failed to save webhook' }));
    } finally {
      setIsSaving(false);
    }
  }, [form, editingWebhook, closeModal, fetchWebhooks, intl]);

  const openDeleteModal = useCallback((webhook: Webhook) => {
    setWebhookToDelete(webhook);
    setIsDeleteModalOpen(true);
  }, []);

  const handleDelete = useCallback(async () => {
    if (!webhookToDelete) return;
    setDeletingId(webhookToDelete.webhook_id);
    setIsDeleteModalOpen(false);
    try {
      await WebhooksApi.deleteWebhook(webhookToDelete.webhook_id);
      await fetchWebhooks();
    } catch (e: any) {
      setError(e?.message ?? intl.formatMessage({ defaultMessage: 'Failed to delete webhook' }));
    } finally {
      setDeletingId(null);
      setWebhookToDelete(null);
    }
  }, [webhookToDelete, fetchWebhooks, intl]);

  const handleTest = useCallback(
    async (webhook: Webhook) => {
      setTestingId(webhook.webhook_id);
      setTestResult(null);
      try {
        const response = await WebhooksApi.testWebhook(webhook.webhook_id);
        const result = response?.result;
        setTestResult({
          webhookId: webhook.webhook_id,
          success: result?.success ?? false,
          message: result?.success
            ? intl.formatMessage(
                { defaultMessage: 'Test succeeded (HTTP {status})' },
                { status: result?.response_status ?? '' },
              )
            : (result?.error_message ?? intl.formatMessage({ defaultMessage: 'Test failed with no error message' })),
        });
      } catch (e: any) {
        setTestResult({
          webhookId: webhook.webhook_id,
          success: false,
          message: e?.message ?? intl.formatMessage({ defaultMessage: 'Failed to test webhook' }),
        });
      } finally {
        setTestingId(null);
      }
    },
    [intl],
  );

  const formatEventLabel = (entity: string, action: string) => {
    const key = eventKey(entity, action);
    const descriptor = eventLabels[key as keyof typeof eventLabels];
    return descriptor ? intl.formatMessage(descriptor) : key;
  };

  const formatEvents = (events: WebhookEvent[]) =>
    events.map((e) => formatEventLabel(e.entity, e.action)).join(', ');

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <Typography.Title level={4} withoutMargins>
            <FormattedMessage defaultMessage="Webhooks" description="Webhooks settings section title" />
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Manage webhooks to receive HTTP notifications when events occur in MLflow."
              description="Webhooks settings section description"
            />
          </Typography.Text>
        </div>
        <Button componentId="mlflow.settings.webhooks.create-button" type="primary" onClick={openCreateModal}>
          <FormattedMessage defaultMessage="Create webhook" description="Create webhook button" />
        </Button>
      </div>

      {error && (
        <Alert
          componentId="mlflow.settings.webhooks.error-alert"
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
        />
      )}

      {testResult && (
        <Alert
          componentId="mlflow.settings.webhooks.test-result-alert"
          type={testResult.success ? 'info' : 'error'}
          message={testResult.message}
          closable
          onClose={() => setTestResult(null)}
        />
      )}

      {loading ? (
        <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
          <Spinner />
        </div>
      ) : webhooks.length === 0 ? (
        <div
          css={{
            padding: theme.spacing.lg,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.legacyBorders.borderRadiusMd,
            textAlign: 'center',
          }}
        >
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="No webhooks configured. Create one to get started."
              description="Empty state for webhooks list"
            />
          </Typography.Text>
        </div>
      ) : (
        <div
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.legacyBorders.borderRadiusMd,
            overflow: 'hidden',
          }}
        >
          {webhooks.map((webhook, index) => (
            <div
              key={webhook.webhook_id}
              css={{
                padding: theme.spacing.md,
                borderBottom: index < webhooks.length - 1 ? `1px solid ${theme.colors.border}` : 'none',
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
                      ? intl.formatMessage({
                          defaultMessage: 'Active',
                          description: 'Webhook active status',
                        })
                      : intl.formatMessage({
                          defaultMessage: 'Disabled',
                          description: 'Webhook disabled status',
                        })}
                  </span>
                </div>
                <Typography.Text size="sm" color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
                  {webhook.url}
                </Typography.Text>
                <Typography.Text size="sm" color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
                  {formatEvents(webhook.events)}
                </Typography.Text>
              </div>
              <div css={{ display: 'flex', gap: theme.spacing.sm, flexShrink: 0 }}>
                <Button
                  componentId="mlflow.settings.webhooks.test-button"
                  size="small"
                  onClick={() => handleTest(webhook)}
                  loading={testingId === webhook.webhook_id}
                  disabled={testingId !== null || deletingId === webhook.webhook_id}
                >
                  <FormattedMessage defaultMessage="Test" description="Test webhook button" />
                </Button>
                <Button
                  componentId="mlflow.settings.webhooks.edit-button"
                  size="small"
                  onClick={() => openEditModal(webhook)}
                  disabled={deletingId === webhook.webhook_id}
                >
                  <FormattedMessage defaultMessage="Edit" description="Edit webhook button" />
                </Button>
                <Button
                  componentId="mlflow.settings.webhooks.delete-button"
                  size="small"
                  danger
                  onClick={() => openDeleteModal(webhook)}
                  loading={deletingId === webhook.webhook_id}
                  disabled={testingId !== null}
                >
                  <FormattedMessage defaultMessage="Delete" description="Delete webhook button" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create/Edit Modal */}
      <Modal
        componentId="mlflow.settings.webhooks.form-modal"
        title={
          editingWebhook
            ? intl.formatMessage({
                defaultMessage: 'Edit webhook',
                description: 'Edit webhook modal title',
              })
            : intl.formatMessage({
                defaultMessage: 'Create webhook',
                description: 'Create webhook modal title',
              })
        }
        visible={isModalOpen}
        onCancel={closeModal}
        onOk={handleSave}
        okText={
          editingWebhook
            ? intl.formatMessage({
                defaultMessage: 'Save',
                description: 'Save webhook button',
              })
            : intl.formatMessage({
                defaultMessage: 'Create',
                description: 'Create webhook confirm button',
              })
        }
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel webhook form button',
        })}
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
              placeholder={intl.formatMessage({
                defaultMessage: 'My webhook',
                description: 'Webhook name placeholder',
              })}
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
                  ? intl.formatMessage({
                      defaultMessage: 'Active',
                      description: 'Webhook active status label',
                    })
                  : intl.formatMessage({
                      defaultMessage: 'Disabled',
                      description: 'Webhook disabled status label',
                    })
              }
              activeLabel={intl.formatMessage({
                defaultMessage: 'Active',
                description: 'Webhook active label',
              })}
              inactiveLabel={intl.formatMessage({
                defaultMessage: 'Disabled',
                description: 'Webhook disabled label',
              })}
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

      {/* Delete Confirmation Modal */}
      <Modal
        componentId="mlflow.settings.webhooks.delete-modal"
        title={intl.formatMessage({
          defaultMessage: 'Delete webhook',
          description: 'Delete webhook modal title',
        })}
        visible={isDeleteModalOpen}
        onCancel={() => {
          setIsDeleteModalOpen(false);
          setWebhookToDelete(null);
        }}
        onOk={handleDelete}
        okText={intl.formatMessage({
          defaultMessage: 'Delete',
          description: 'Confirm delete webhook button',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel delete webhook button',
        })}
        okButtonProps={{ danger: true }}
      >
        <Typography.Text>
          <FormattedMessage
            defaultMessage='Are you sure you want to delete the webhook "{name}"? This action cannot be undone.'
            description="Delete webhook confirmation message"
            values={{ name: webhookToDelete?.name ?? '' }}
          />
        </Typography.Text>
      </Modal>
    </div>
  );
};

export default WebhooksSettings;

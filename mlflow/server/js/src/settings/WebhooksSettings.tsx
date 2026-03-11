import { useCallback, useEffect, useState } from 'react';
import { Alert, Button, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { WebhooksApi } from './webhooksApi';
import type { Webhook } from './webhooksApi';
import WebhookListItem from './WebhookListItem';
import WebhookFormModal from './WebhookFormModal';
import WebhookDeleteModal from './WebhookDeleteModal';

interface WebhooksComponentIds {
  createButton: string;
  errorAlert: string;
  testResultAlert: string;
}

const DEFAULT_COMPONENT_IDS: WebhooksComponentIds = {
  createButton: 'mlflow.settings.webhooks.create-button',
  errorAlert: 'mlflow.settings.webhooks.error-alert',
  testResultAlert: 'mlflow.settings.webhooks.test-result-alert',
};

interface WebhooksSettingsProps {
  /** Filter displayed webhooks to only those containing at least one event whose entity matches this value exactly */
  eventFilter?: string;
  /** Title override */
  title?: React.ReactNode;
  /** Description override */
  description?: React.ReactNode;
  /** Static componentIds for namespacing */
  componentIds?: WebhooksComponentIds;
}

const WebhooksSettings = ({
  eventFilter,
  title,
  description,
  componentIds = DEFAULT_COMPONENT_IDS,
}: WebhooksSettingsProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [webhooks, setWebhooks] = useState<Webhook[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingWebhook, setEditingWebhook] = useState<Webhook | null>(null);

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
      const all = response?.webhooks ?? [];
      if (eventFilter) {
        setWebhooks(all.filter((w) => w.events.some((e) => e.entity === eventFilter)));
      } else {
        setWebhooks(all);
      }
    } catch (e: any) {
      setError(e?.message ?? intl.formatMessage({ defaultMessage: 'Failed to load webhooks' }));
    } finally {
      setLoading(false);
    }
  }, [intl, eventFilter]);

  useEffect(() => {
    fetchWebhooks();
  }, [fetchWebhooks]);

  const openCreateModal = useCallback(() => {
    setEditingWebhook(null);
    setIsModalOpen(true);
  }, []);

  const openEditModal = useCallback((webhook: Webhook) => {
    setEditingWebhook(webhook);
    setIsModalOpen(true);
  }, []);

  const closeModal = useCallback(() => {
    setIsModalOpen(false);
    setEditingWebhook(null);
  }, []);

  const handleSaved = useCallback(async () => {
    closeModal();
    await fetchWebhooks();
  }, [closeModal, fetchWebhooks]);

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
          message: e?.message ?? intl.formatMessage({ defaultMessage: 'Failed to invoke webhook' }),
        });
      } finally {
        setTestingId(null);
      }
    },
    [intl],
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <Typography.Title level={4} withoutMargins>
            {title ?? <FormattedMessage defaultMessage="Webhooks" description="Webhooks settings section title" />}
          </Typography.Title>
          <Typography.Text color="secondary">
            {description ?? (
              <FormattedMessage
                defaultMessage="Manage webhooks to receive HTTP notifications when events occur in MLflow."
                description="Webhooks settings section description"
              />
            )}
          </Typography.Text>
        </div>
        <Button componentId={componentIds.createButton} type="primary" onClick={openCreateModal}>
          <FormattedMessage defaultMessage="Create webhook" description="Create webhook button" />
        </Button>
      </div>

      {error && (
        <Alert
          componentId={componentIds.errorAlert}
          type="error"
          message={error}
          closable
          onClose={() => setError(null)}
        />
      )}

      {testResult && (
        <Alert
          componentId={componentIds.testResultAlert}
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
              defaultMessage="No webhooks configured. Create one to get started. <link>Learn more about webhooks.</link>"
              description="Empty state for webhooks list"
              values={{
                link: (chunks: any) => (
                  <a href="https://mlflow.org/docs/latest/ml/webhooks/" target="_blank" rel="noopener noreferrer">
                    {chunks}
                  </a>
                ),
              }}
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
            <WebhookListItem
              key={webhook.webhook_id}
              webhook={webhook}
              isLast={index === webhooks.length - 1}
              testingId={testingId}
              deletingId={deletingId}
              onTest={handleTest}
              onEdit={openEditModal}
              onDelete={openDeleteModal}
            />
          ))}
        </div>
      )}

      {isModalOpen && (
        <WebhookFormModal
          visible={isModalOpen}
          editingWebhook={editingWebhook}
          onClose={closeModal}
          onSaved={handleSaved}
          eventFilter={eventFilter}
        />
      )}

      <WebhookDeleteModal
        visible={isDeleteModalOpen}
        webhook={webhookToDelete}
        onCancel={() => {
          setIsDeleteModalOpen(false);
          setWebhookToDelete(null);
        }}
        onConfirm={handleDelete}
      />
    </div>
  );
};

export default WebhooksSettings;

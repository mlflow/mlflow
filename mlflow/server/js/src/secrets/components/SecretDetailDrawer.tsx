import {
  Button,
  DangerModal,
  Drawer,
  LightningIcon,
  PencilIcon,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useCallback, useState } from 'react';
import type { Secret } from '../types';
import { SecretBindingsList } from './SecretBindingsList';
import { useUnbindSecretMutation } from '../hooks/useUnbindSecretMutation';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { Descriptions } from '@mlflow/mlflow/src/common/components/Descriptions';

export interface SecretDetailDrawerProps {
  secret: Secret | null;
  open: boolean;
  onClose: () => void;
  onUpdate?: (secret: Secret) => void;
  onDelete?: (secret: Secret) => void;
}

export const SecretDetailDrawer = ({ secret, open, onClose, onUpdate, onDelete }: SecretDetailDrawerProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [unbindingId, setUnbindingId] = useState<string | null>(null);

  const { unbindSecret, isLoading: isUnbinding } = useUnbindSecretMutation({
    onSuccess: () => {
      setUnbindingId(null);
    },
    onError: (error: Error) => {
      console.error('Failed to unbind secret:', error);
      setUnbindingId(null);
    },
  });

  const handleUnbind = useCallback((bindingId: string) => {
    setUnbindingId(bindingId);
  }, []);

  const confirmUnbind = useCallback(() => {
    if (unbindingId) {
      unbindSecret({ binding_id: unbindingId });
    }
  }, [unbindingId, unbindSecret]);

  return (
    <Drawer.Root modal open={open} onOpenChange={onClose}>
      <Drawer.Content
        componentId="mlflow.secrets.detail_drawer"
        width="700px"
        title={
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 32,
                height: 32,
                borderRadius: theme.borders.borderRadiusMd,
                backgroundColor: theme.colors.backgroundSecondary,
              }}
            >
              <LightningIcon css={{ fontSize: 18 }} />
            </div>
            <FormattedMessage
              defaultMessage="Secret Details"
              description="Secret detail drawer > drawer title"
            />
          </div>
        }
      >
        {secret && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
            {/* Header section with name and status */}
            <div
              css={{
                padding: theme.spacing.md,
                borderRadius: theme.borders.borderRadiusMd,
                backgroundColor: theme.colors.backgroundSecondary,
                border: `1px solid ${theme.colors.border}`,
              }}
            >
              <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: theme.spacing.sm }}>
                <Typography.Title level={3} css={{ margin: 0 }}>
                  {secret.secret_name}
                </Typography.Title>
                {secret.is_shared && (
                  <Tag componentId="mlflow.secrets.detail_drawer.shared_tag" color="lime">
                    <FormattedMessage
                      defaultMessage="Shared"
                      description="Secret detail drawer > shared tag"
                    />
                  </Tag>
                )}
              </div>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage
                    defaultMessage="Masked Value:"
                    description="Secret detail drawer > masked value prefix"
                  />
                </Typography.Text>
                <Typography.Text css={{ fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm }}>
                  {secret.masked_value}
                </Typography.Text>
              </div>
            </div>

            {/* Action buttons */}
            <div css={{ display: 'flex', gap: theme.spacing.sm }}>
              <Button
                componentId="mlflow.secrets.detail_drawer.update_button"
                icon={<PencilIcon />}
                onClick={() => {
                  onUpdate?.(secret);
                }}
              >
                <FormattedMessage
                  defaultMessage="Update Secret"
                  description="Secret detail drawer > update button"
                />
              </Button>
              <Button
                componentId="mlflow.secrets.detail_drawer.delete_button"
                icon={<TrashIcon />}
                danger
                onClick={() => {
                  onDelete?.(secret);
                }}
              >
                <FormattedMessage
                  defaultMessage="Delete Secret"
                  description="Secret detail drawer > delete button"
                />
              </Button>
            </div>

            {/* Metadata section */}
            <div>
              <Typography.Title level={4} css={{ margin: 0, marginBottom: theme.spacing.md }}>
                <FormattedMessage
                  defaultMessage="Metadata"
                  description="Secret detail drawer > metadata section title"
                />
              </Typography.Title>
              <Descriptions columns={1}>
                {secret.created_by && (
                  <Descriptions.Item
                    label={
                      <FormattedMessage
                        defaultMessage="Created By"
                        description="Secret detail drawer > created by label"
                      />
                    }
                  >
                    <Typography.Text>{secret.created_by}</Typography.Text>
                  </Descriptions.Item>
                )}
                <Descriptions.Item
                  label={
                    <FormattedMessage
                      defaultMessage="Created At"
                      description="Secret detail drawer > created at label"
                    />
                  }
                >
                  <Typography.Text>{Utils.formatTimestamp(secret.created_at)}</Typography.Text>
                </Descriptions.Item>
                <Descriptions.Item
                  label={
                    <FormattedMessage
                      defaultMessage="Last Updated"
                      description="Secret detail drawer > last updated label"
                    />
                  }
                >
                  <Typography.Text>{Utils.formatTimestamp(secret.last_updated_at)}</Typography.Text>
                </Descriptions.Item>
              </Descriptions>
            </div>

            {/* Bindings section */}
            <div>
              <Typography.Title level={4} css={{ marginBottom: theme.spacing.md }}>
                <FormattedMessage
                  defaultMessage="Bindings"
                  description="Secret detail drawer > bindings section title"
                />
              </Typography.Title>
              <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                <FormattedMessage
                  defaultMessage="These resources are currently using this secret and will have access to its value through the specified environment variable."
                  description="Secret detail drawer > bindings description"
                />
              </Typography.Text>
              <SecretBindingsList
                secretId={secret.secret_id}
                variant="default"
                isSharedSecret={secret.is_shared}
                onUnbind={handleUnbind}
              />
            </div>
          </div>
        )}

        <DangerModal
          componentId="mlflow.secrets.detail_drawer.unbind_modal"
          visible={!!unbindingId}
          onCancel={() => setUnbindingId(null)}
          okText={intl.formatMessage({
            defaultMessage: 'Unbind',
            description: 'Unbind confirmation modal > unbind button text',
          })}
          cancelText={intl.formatMessage({
            defaultMessage: 'Cancel',
            description: 'Unbind confirmation modal > cancel button text',
          })}
          onOk={confirmUnbind}
          okButtonProps={{ loading: isUnbinding }}
          title={
            <FormattedMessage
              defaultMessage="Unbind Secret"
              description="Unbind confirmation modal > modal title"
            />
          }
        >
          <Typography.Text>
            <FormattedMessage
              defaultMessage="Are you sure you want to unbind this secret from the resource? The resource will no longer have access to this secret."
              description="Unbind confirmation modal > confirmation message"
            />
          </Typography.Text>
        </DangerModal>
      </Drawer.Content>
    </Drawer.Root>
  );
};

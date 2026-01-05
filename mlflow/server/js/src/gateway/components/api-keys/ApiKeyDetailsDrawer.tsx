import { useCallback } from 'react';
import {
  Button,
  Drawer,
  KeyIcon,
  PencilIcon,
  Spacer,
  TrashIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { SecretDetails } from '../secrets';
import type { SecretInfo } from '../../types';

interface ApiKeyDetailsDrawerProps {
  open: boolean;
  secret: SecretInfo | null;
  onClose: () => void;
  onEdit?: (secret: SecretInfo) => void;
  onDelete?: (secret: SecretInfo) => void;
}

export const ApiKeyDetailsDrawer = ({ open, secret, onClose, onEdit, onDelete }: ApiKeyDetailsDrawerProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  const handleOpenChange = useCallback(
    (isOpen: boolean) => {
      if (!isOpen) {
        onClose();
      }
    },
    [onClose],
  );

  const handleEditClick = useCallback(() => {
    if (secret && onEdit) {
      onClose();
      onEdit(secret);
    }
  }, [secret, onEdit, onClose]);

  const handleDeleteClick = useCallback(() => {
    if (secret && onDelete) {
      onClose();
      onDelete(secret);
    }
  }, [secret, onDelete, onClose]);

  return (
    <Drawer.Root modal open={open} onOpenChange={handleOpenChange}>
      <Drawer.Content
        componentId="mlflow.gateway.api-key-details.drawer"
        width={480}
        title={
          <span
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
            }}
          >
            <span
              css={{
                borderRadius: theme.borders.borderRadiusSm,
                background: theme.colors.actionDefaultBackgroundHover,
                padding: theme.spacing.xs,
                color: theme.colors.blue500,
                height: 'min-content',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <KeyIcon />
            </span>
            <FormattedMessage defaultMessage="API Key Details" description="Title for the API key details drawer" />
          </span>
        }
      >
        <Spacer size="md" />
        {secret && (
          <div css={{ display: 'flex', flexDirection: 'column' }}>
            <SecretDetails secret={secret} showCard={false} />
            <div css={{ marginTop: theme.spacing.lg, display: 'flex', gap: theme.spacing.sm }}>
              <Button
                componentId="mlflow.gateway.api-key-details.drawer.edit-button"
                icon={<PencilIcon />}
                onClick={handleEditClick}
                aria-label={formatMessage({
                  defaultMessage: 'Edit API key',
                  description: 'Gateway > API key details drawer > Edit API key button aria label',
                })}
              >
                <FormattedMessage
                  defaultMessage="Edit API Key"
                  description="Gateway > API key details drawer > Edit API key button"
                />
              </Button>
              <Button
                componentId="mlflow.gateway.api-key-details.drawer.delete-button"
                danger
                icon={<TrashIcon />}
                onClick={handleDeleteClick}
                aria-label={formatMessage({
                  defaultMessage: 'Delete API key',
                  description: 'Gateway > API key details drawer > Delete API key button aria label',
                })}
              >
                <FormattedMessage
                  defaultMessage="Delete API Key"
                  description="Gateway > API key details drawer > Delete API key button"
                />
              </Button>
            </div>
          </div>
        )}
      </Drawer.Content>
    </Drawer.Root>
  );
};

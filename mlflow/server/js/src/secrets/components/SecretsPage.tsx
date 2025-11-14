import {
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  GearIcon,
  Header,
  Input,
  Notification,
  SearchIcon,
  Spacer,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useState, useCallback, useMemo } from 'react';
import type { SortingState } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { useListSecrets } from '../hooks/useListSecrets';
import { SecretsTable } from './SecretsTable';
import { CreateSecretModal } from './CreateSecretModal';
import { UpdateSecretModal } from './UpdateSecretModal';
import { DeleteSecretModal } from './DeleteSecretModal';
import { SecretDetailDrawer } from './SecretDetailDrawer';
import { SecretManagementDrawer } from './SecretManagementDrawer';
import type { Secret } from '../types';

export default function SecretsPage() {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { secrets = [], isLoading, error } = useListSecrets({ enabled: true });

  const [sorting, setSorting] = useState<SortingState>([{ id: 'created_at', desc: true }]);
  const [hiddenColumns, setHiddenColumns] = useState<string[]>([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showUpdateModal, setShowUpdateModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showManagementDrawer, setShowManagementDrawer] = useState(false);
  const [selectedSecret, setSelectedSecret] = useState<Secret | null>(null);
  const [searchText, setSearchText] = useState('');
  const [isSharedFilter, setIsSharedFilter] = useState<'all' | 'shared' | 'private'>('all');
  const [showSuccessNotification, setShowSuccessNotification] = useState(false);
  const [updatedSecretName, setUpdatedSecretName] = useState('');

  const handleCreateSecret = useCallback(() => {
    setShowCreateModal(true);
  }, []);

  const handleSecretClicked = useCallback((secret: Secret) => {
    setSelectedSecret(secret);
    setShowDetailModal(true);
  }, []);

  const handleUpdateSecret = useCallback((secret: Secret) => {
    setSelectedSecret(secret);
    setShowUpdateModal(true);
    setShowDetailModal(false);
  }, []);

  const handleUpdateSuccess = useCallback((secretName: string) => {
    setUpdatedSecretName(secretName);
    setShowSuccessNotification(true);
    setTimeout(() => setShowSuccessNotification(false), 3000);
  }, []);

  const handleDeleteSecret = useCallback((secret: Secret) => {
    setSelectedSecret(secret);
    setShowDeleteModal(true);
    setShowDetailModal(false);
  }, []);

  const toggleHiddenColumn = useCallback((columnId: string) => {
    setHiddenColumns((prev) => {
      if (prev.includes(columnId)) {
        return prev.filter((id) => id !== columnId);
      }
      return [...prev, columnId];
    });
  }, []);

  const filteredSecrets = useMemo(() => {
    return secrets.filter((secret) => {
      const matchesSearch = searchText ? secret.secret_name.toLowerCase().includes(searchText.toLowerCase()) : true;

      const matchesSharedFilter =
        isSharedFilter === 'all' ||
        (isSharedFilter === 'shared' && secret.is_shared) ||
        (isSharedFilter === 'private' && !secret.is_shared);

      return matchesSearch && matchesSharedFilter;
    });
  }, [secrets, searchText, isSharedFilter]);

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={<FormattedMessage defaultMessage="Secrets" description="Header title for the secrets management page" />}
        breadcrumbs={[]}
        buttons={
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Button
              componentId="mlflow.secrets.manage_secrets_button"
              icon={<GearIcon />}
              onClick={() => setShowManagementDrawer(true)}
            >
              <FormattedMessage defaultMessage="Manage Secrets" description="Manage secrets button label" />
            </Button>
            <Button componentId="mlflow.secrets.create_secret_button" type="primary" onClick={handleCreateSecret}>
              <FormattedMessage defaultMessage="Create Secret" description="Create secret button label" />
            </Button>
          </div>
        }
      />
      <Spacer shrinks={false} />
      <div css={{ padding: theme.spacing.lg, paddingBottom: theme.spacing.sm }}>
        <div css={{ display: 'flex', gap: theme.spacing.md, alignItems: 'center' }}>
          <Input
            componentId="mlflow.secrets.search_input"
            placeholder={intl.formatMessage({
              defaultMessage: 'Search secrets by name',
              description: 'Secrets page > search input placeholder',
            })}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            prefix={<SearchIcon />}
            allowClear
            onClear={() => setSearchText('')}
            css={{ width: 400 }}
          />
          <DialogCombobox
            componentId="mlflow.secrets.shared_filter"
            label={intl.formatMessage({
              defaultMessage: 'Shared',
              description: 'Secrets page > shared filter label',
            })}
            value={[isSharedFilter]}
          >
            <DialogComboboxTrigger allowClear={false} />
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                <DialogComboboxOptionListSelectItem
                  checked={isSharedFilter === 'all'}
                  value="all"
                  onChange={() => setIsSharedFilter('all')}
                >
                  <FormattedMessage defaultMessage="All" description="Secrets page > shared filter all option" />
                </DialogComboboxOptionListSelectItem>
                <DialogComboboxOptionListSelectItem
                  checked={isSharedFilter === 'shared'}
                  value="shared"
                  onChange={() => setIsSharedFilter('shared')}
                >
                  <FormattedMessage defaultMessage="Shared" description="Secrets page > shared filter shared option" />
                </DialogComboboxOptionListSelectItem>
                <DialogComboboxOptionListSelectItem
                  checked={isSharedFilter === 'private'}
                  value="private"
                  onChange={() => setIsSharedFilter('private')}
                >
                  <FormattedMessage
                    defaultMessage="Private"
                    description="Secrets page > shared filter private option"
                  />
                </DialogComboboxOptionListSelectItem>
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        </div>
      </div>
      <div
        css={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          padding: theme.spacing.lg,
          paddingTop: 0,
        }}
      >
        <SecretsTable
          secrets={filteredSecrets}
          loading={isLoading}
          error={error ?? undefined}
          onSecretClicked={handleSecretClicked}
          onUpdateSecret={handleUpdateSecret}
          onDeleteSecret={handleDeleteSecret}
          sorting={sorting}
          setSorting={setSorting}
          hiddenColumns={hiddenColumns}
          toggleHiddenColumn={toggleHiddenColumn}
        />
      </div>

      <CreateSecretModal visible={showCreateModal} onCancel={() => setShowCreateModal(false)} />
      <UpdateSecretModal
        secret={selectedSecret}
        visible={showUpdateModal}
        onCancel={() => {
          setShowUpdateModal(false);
          setSelectedSecret(null);
        }}
        onSuccess={handleUpdateSuccess}
      />
      <DeleteSecretModal
        secret={selectedSecret}
        visible={showDeleteModal}
        onCancel={() => {
          setShowDeleteModal(false);
          setSelectedSecret(null);
        }}
      />
      <SecretDetailDrawer
        secret={selectedSecret}
        open={showDetailModal}
        onClose={() => {
          setShowDetailModal(false);
          setSelectedSecret(null);
        }}
        onUpdate={handleUpdateSecret}
        onDelete={handleDeleteSecret}
      />

      {showSuccessNotification && (
        <Notification.Provider>
          <Notification.Root severity="success" componentId="mlflow.secrets.success_notification">
            <Notification.Title>
              <FormattedMessage
                defaultMessage="Secret {secretName} updated successfully"
                description="Success notification message for updating secret"
                values={{ secretName: updatedSecretName }}
              />
            </Notification.Title>
          </Notification.Root>
          <Notification.Viewport />
        </Notification.Provider>
      )}

      <SecretManagementDrawer open={showManagementDrawer} onClose={() => setShowManagementDrawer(false)} />
    </ScrollablePageWrapper>
  );
}

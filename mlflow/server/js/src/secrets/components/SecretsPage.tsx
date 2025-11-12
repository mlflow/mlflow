import {
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
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
import { UpdateApiKeyModal } from './UpdateApiKeyModal';
import { UpdateModelModal } from './UpdateModelModal';
import { DeleteSecretModal } from './DeleteSecretModal';
import { SecretDetailDrawer } from './SecretDetailDrawer';
import type { Secret } from '../types';

export default function SecretsPage() {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { secrets = [], isLoading, error } = useListSecrets({ enabled: true });

  const [sorting, setSorting] = useState<SortingState>([{ id: 'created_at', desc: true }]);
  const [hiddenColumns, setHiddenColumns] = useState<string[]>([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showUpdateApiKeyModal, setShowUpdateApiKeyModal] = useState(false);
  const [showUpdateModelModal, setShowUpdateModelModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showDetailModal, setShowDetailModal] = useState(false);
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

  const handleUpdateApiKey = useCallback((secret: Secret) => {
    setSelectedSecret(secret);
    setShowUpdateApiKeyModal(true);
    setShowDetailModal(false);
  }, []);

  const handleUpdateModel = useCallback((secret: Secret) => {
    setSelectedSecret(secret);
    setShowUpdateModelModal(true);
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
      const matchesSearch = searchText
        ? secret.secret_name.toLowerCase().includes(searchText.toLowerCase())
        : true;

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
        title={<FormattedMessage defaultMessage="Gateway" description="Header title for the gateway management page" />}
        breadcrumbs={[]}
        buttons={
          <Button componentId="mlflow.secrets.create_secret_button" type="primary" onClick={handleCreateSecret}>
            <FormattedMessage defaultMessage="Add Model" description="Add model button label" />
          </Button>
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
                  <FormattedMessage defaultMessage="Private" description="Secrets page > shared filter private option" />
                </DialogComboboxOptionListSelectItem>
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        </div>
      </div>
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: theme.spacing.lg, paddingTop: 0 }}>
        <SecretsTable
          secrets={filteredSecrets}
          loading={isLoading}
          error={error ?? undefined}
          onSecretClicked={handleSecretClicked}
          onUpdateSecret={handleUpdateApiKey}
          onUpdateModel={handleUpdateModel}
          onDeleteSecret={handleDeleteSecret}
          sorting={sorting}
          setSorting={setSorting}
          hiddenColumns={hiddenColumns}
          toggleHiddenColumn={toggleHiddenColumn}
        />
      </div>

      <CreateSecretModal visible={showCreateModal} onCancel={() => setShowCreateModal(false)} />
      <UpdateApiKeyModal
        secret={selectedSecret}
        visible={showUpdateApiKeyModal}
        onCancel={() => {
          setShowUpdateApiKeyModal(false);
          setSelectedSecret(null);
        }}
        onSuccess={handleUpdateSuccess}
      />
      <UpdateModelModal
        secret={selectedSecret}
        visible={showUpdateModelModal}
        onCancel={() => {
          setShowUpdateModelModal(false);
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
        onUpdateApiKey={handleUpdateApiKey}
        onUpdateModel={handleUpdateModel}
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
    </ScrollablePageWrapper>
  );
}

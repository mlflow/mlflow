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
  Spinner,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useState, useCallback, useMemo } from 'react';
import { useDebounce } from 'use-debounce';
import type { SortingState } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { useBackendSupport } from '@mlflow/mlflow/src/common/hooks/useBackendSupport';
import { useListSecrets } from '../hooks/useListSecrets';
import { SecretsTable } from './SecretsTable';
import { useCreateSecretModal } from '../hooks/modals/useCreateSecretModal';
import { useUpdateSecretModal } from '../hooks/modals/useUpdateSecretModal';
import { useDeleteSecretModal } from '../hooks/modals/useDeleteSecretModal';
import { SecretDetailDrawer } from './SecretDetailDrawer';
import { SecretManagementDrawer } from './SecretManagementDrawer';
import { RouteDetailDrawer } from './RouteDetailDrawer';
import { GatewayRequiresSqlBackend } from './GatewayRequiresSqlBackend';
import type { Secret, Endpoint } from '../types';
import { useListEndpoints } from '../hooks/useListEndpoints';

export default function SecretsPage() {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { isSqlBackend, storeType, isLoading: isCheckingBackend } = useBackendSupport();
  const { secrets = [], isLoading, error } = useListSecrets({ enabled: isSqlBackend === true });
  const { endpoints = [] } = useListEndpoints({ enabled: isSqlBackend === true });

  const [sorting, setSorting] = useState<SortingState>([{ id: 'created_at', desc: true }]);
  const [hiddenColumns, setHiddenColumns] = useState<string[]>([]);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showManagementDrawer, setShowManagementDrawer] = useState(false);
  const [showEndpointDrawer, setShowEndpointDrawer] = useState(false);
  const [selectedSecret, setSelectedSecret] = useState<Secret | null>(null);
  const [selectedEndpoint, setSelectedEndpoint] = useState<Endpoint | null>(null);
  const [searchText, setSearchText] = useState('');
  const [debouncedSearchText] = useDebounce(searchText, 500);
  const [isSharedFilter, setIsSharedFilter] = useState<'all' | 'shared' | 'private'>('all');
  const [showSuccessNotification, setShowSuccessNotification] = useState(false);
  const [updatedSecretName, setUpdatedSecretName] = useState('');

  // Modal hooks
  const { CreateSecretModal, openModal: openCreateModal } = useCreateSecretModal({
    onSuccess: () => {
      // Refetch is handled automatically by React Query invalidation
    },
  });

  const { UpdateSecretModal, openModal: openUpdateModal } = useUpdateSecretModal({
    secret: selectedSecret,
    onSuccess: (secretName) => {
      setUpdatedSecretName(secretName);
      setShowSuccessNotification(true);
      setTimeout(() => setShowSuccessNotification(false), 3000);
    },
  });

  const { DeleteSecretModal, openModal: openDeleteModal } = useDeleteSecretModal({
    secret: selectedSecret,
    onSuccess: () => {
      setShowDetailModal(false);
    },
  });

  const handleCreateSecret = useCallback(() => {
    openCreateModal();
  }, [openCreateModal]);

  const handleSecretClicked = useCallback((secret: Secret) => {
    setSelectedSecret(secret);
    setShowDetailModal(true);
  }, []);

  const handleUpdateSecret = useCallback(
    (secret: Secret) => {
      setSelectedSecret(secret);
      setShowDetailModal(false);
      openUpdateModal();
    },
    [openUpdateModal],
  );

  const handleDeleteSecret = useCallback(
    (secret: Secret) => {
      setSelectedSecret(secret);
      setShowDetailModal(false);
      openDeleteModal();
    },
    [openDeleteModal],
  );

  const toggleHiddenColumn = useCallback((columnId: string) => {
    setHiddenColumns((prev) => {
      if (prev.includes(columnId)) {
        return prev.filter((id) => id !== columnId);
      }
      return [...prev, columnId];
    });
  }, []);

  const handleEndpointClick = useCallback((endpoint: Endpoint) => {
    // Close management drawer first
    setShowManagementDrawer(false);
    // Set selected endpoint and open drawer after brief delay
    setTimeout(() => {
      setSelectedEndpoint(endpoint);
      setShowEndpointDrawer(true);
    }, 100);
  }, []);

  const filteredSecrets = useMemo(() => {
    return secrets.filter((secret) => {
      const matchesSearch = debouncedSearchText
        ? secret.secret_name.toLowerCase().includes(debouncedSearchText.toLowerCase())
        : true;

      const matchesSharedFilter =
        isSharedFilter === 'all' ||
        (isSharedFilter === 'shared' && secret.is_shared) ||
        (isSharedFilter === 'private' && !secret.is_shared);

      return matchesSearch && matchesSharedFilter;
    });
  }, [secrets, debouncedSearchText, isSharedFilter]);

  // Show placeholder if FileStore is detected
  if (isCheckingBackend) {
    return (
      <ScrollablePageWrapper>
        <div css={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
          <Spinner />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (isSqlBackend === false) {
    return <GatewayRequiresSqlBackend storeType={storeType} />;
  }

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

      {CreateSecretModal}
      {UpdateSecretModal}
      {DeleteSecretModal}
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

      <SecretManagementDrawer
        open={showManagementDrawer}
        onClose={() => setShowManagementDrawer(false)}
        onEndpointClick={handleEndpointClick}
      />

      <RouteDetailDrawer
        route={selectedEndpoint}
        open={showEndpointDrawer}
        onClose={() => {
          setShowEndpointDrawer(false);
          setSelectedEndpoint(null);
        }}
      />
    </ScrollablePageWrapper>
  );
}

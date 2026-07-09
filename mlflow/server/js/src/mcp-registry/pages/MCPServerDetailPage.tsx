import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  DropdownMenu,
  Header,
  OverflowIcon,
  Spacer,
  Spinner,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { Link, useNavigate, useParams, useSearchParams } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ConfirmationModal } from '../../admin/ConfirmationModal';
import { useEditAliasesModal } from '../../common/hooks/useEditAliasesModal';
import MCPRegistryRoutes from '../routes';
import { MCPRegistryApi } from '../api';
import {
  useMCPServerQuery,
  useMCPServerVersionsQuery,
  useLatestMCPServerVersionQuery,
} from '../hooks/useMCPServerDetailQuery';
import { useDeleteMCPServer, useUpdateMCPServerDisplayName } from '../hooks/useMCPServerVersionMutations';
import { useCreateMCPServerVersionModal } from '../hooks/useCreateMCPServerVersionModal';
import { useUpdateMCPServerVersionMetadataModal } from '../hooks/useUpdateMCPServerVersionMetadataModal';
import { MCPServerVersionList } from '../components/MCPServerVersionList';
import { MCPServerVersionDetail } from '../components/MCPServerVersionDetail';
import { UpdateVersionDisplayNameModal } from '../components/UpdateVersionDisplayNameModal';
import { MCPServerTagsBox } from '../components/MCPServerTagsBox';
import { LATEST_ALIAS, resolveDisplayName } from '../utils';

const getAliasesModalTitle = (version: string) => (
  <FormattedMessage
    defaultMessage="Add/edit alias for MCP server version {version}"
    description="Title for the edit aliases modal on the MCP server detail page"
    values={{ version }}
  />
);

const MCPServerDetailPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const params = useParams<{ serverName: string }>();
  const [searchParams] = useSearchParams();
  const versionFromUrl = searchParams.get('version') ?? undefined;
  const serverName = decodeURIComponent(params.serverName ?? '');
  const [selectedVersion, setSelectedVersion] = useState<string | undefined>(undefined);
  const [deleteServerModalVisible, setDeleteServerModalVisible] = useState(false);
  const [editServerDisplayNameVisible, setEditServerDisplayNameVisible] = useState(false);
  const deleteServerMutation = useDeleteMCPServer();
  const updateDisplayNameMutation = useUpdateMCPServerDisplayName(serverName);
  const {
    data: server,
    isLoading: serverLoading,
    error: serverError,
    refetch: refetchServer,
  } = useMCPServerQuery(serverName);
  const {
    data: versions,
    isLoading: versionsLoading,
    error: versionsError,
    refetch: refetchVersions,
  } = useMCPServerVersionsQuery(serverName);
  const { data: latestVersion, refetch: refetchLatestVersion } = useLatestMCPServerVersionQuery(serverName);

  useEffect(() => {
    if (!versions?.length) {
      setSelectedVersion(undefined);
      return;
    }
    setSelectedVersion((prev) => {
      const currentStillValid = prev && versions.some((v) => v.version === prev);
      if (currentStillValid) return prev;
      const urlVersion =
        versionFromUrl && versions.some((v) => v.version === versionFromUrl) ? versionFromUrl : undefined;
      return urlVersion ?? versions[0].version;
    });
  }, [versions, versionFromUrl]);

  const currentVersion = versions?.find((v) => v.version === selectedVersion);

  const resolvedLatestVersion = latestVersion?.version;

  const aliasesByVersion = useMemo(() => {
    const result: Record<string, string[]> = {};
    server?.aliases?.forEach(({ alias, version }) => {
      if (!result[version]) {
        result[version] = [];
      }
      result[version].push(alias);
    });
    if (resolvedLatestVersion) {
      if (!result[resolvedLatestVersion]) {
        result[resolvedLatestVersion] = [];
      }
      if (!result[resolvedLatestVersion].includes(LATEST_ALIAS)) {
        result[resolvedLatestVersion].unshift(LATEST_ALIAS);
      }
    }
    return result;
  }, [server?.aliases, resolvedLatestVersion]);

  const refetchAll = useCallback(async () => {
    await Promise.all([refetchServer(), refetchVersions(), refetchLatestVersion()]);
  }, [refetchServer, refetchVersions, refetchLatestVersion]);

  const { CreateMCPServerVersionModal, openModal: openCreateVersionModal } = useCreateMCPServerVersionModal({
    serverName: serverName,
    latestVersion: versions?.[0],
    onSuccess: async ({ version }) => {
      await refetchAll();
      setSelectedVersion(version);
    },
  });

  const { EditAliasesModal, showEditAliasesModal } = useEditAliasesModal({
    aliases: server?.aliases ?? [],
    onSuccess: refetchAll,
    getTitle: getAliasesModalTitle,
    onSave: async (_currentlyEditedVersion: string, existingAliases: string[], draftAliases: string[]) => {
      const addedAliases = draftAliases.filter((a) => !existingAliases.includes(a));
      const deletedAliases = existingAliases.filter((a) => !draftAliases.includes(a));
      await Promise.all([
        ...addedAliases.map((alias) =>
          MCPRegistryApi.setMCPServerAlias(serverName, { alias, version: _currentlyEditedVersion }),
        ),
        ...deletedAliases.map((alias) => MCPRegistryApi.deleteMCPServerAlias(serverName, alias)),
      ]);
    },
    description: (
      <FormattedMessage
        defaultMessage="Aliases allow you to assign a mutable, named reference to a particular MCP server version."
        description="Description for the edit aliases modal on the MCP server detail page"
      />
    ),
  });

  const { EditMCPServerVersionMetadataModal, showEditMetadataModal } = useUpdateMCPServerVersionMetadataModal({
    serverName: serverName,
    onSuccess: refetchAll,
  });

  const breadcrumbs = (
    <Breadcrumb>
      <Breadcrumb.Item>
        <Link componentId="mlflow.mcp_registry.detail.breadcrumb_back" to={MCPRegistryRoutes.mcpRegistryPageRoute}>
          <FormattedMessage defaultMessage="MCP Registry" description="MCP Registry breadcrumb link" />
        </Link>
      </Breadcrumb.Item>
    </Breadcrumb>
  );

  if (serverLoading) {
    return (
      <ScrollablePageWrapper>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            minHeight: 400,
          }}
        >
          <Spinner size="small" />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (serverError || !server) {
    return (
      <ScrollablePageWrapper>
        <Spacer shrinks={false} />
        <Header breadcrumbs={breadcrumbs} title="" />
        <Alert
          componentId="mlflow.mcp_registry.detail.error"
          type="error"
          message={
            <FormattedMessage
              defaultMessage="Failed to load MCP server"
              description="MCP server detail page error title"
            />
          }
          description={serverError?.message}
          closable={false}
        />
      </ScrollablePageWrapper>
    );
  }

  const displayName = resolveDisplayName(server);

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        breadcrumbs={breadcrumbs}
        title={displayName}
        buttons={
          <>
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <Button
                  componentId="mlflow.mcp_registry.detail.actions"
                  icon={<OverflowIcon />}
                  aria-label={intl.formatMessage({
                    defaultMessage: 'More actions',
                    description: 'Aria label for MCP server detail actions overflow menu',
                  })}
                />
              </DropdownMenu.Trigger>
              <DropdownMenu.Content>
                <DropdownMenu.Item
                  componentId="mlflow.mcp_registry.detail.actions.edit_display_name"
                  onClick={() => setEditServerDisplayNameVisible(true)}
                >
                  <FormattedMessage
                    defaultMessage="Edit display name"
                    description="MCP server detail edit server display name action"
                  />
                </DropdownMenu.Item>
                <DropdownMenu.Item
                  componentId="mlflow.mcp_registry.detail.actions.delete"
                  onClick={() => setDeleteServerModalVisible(true)}
                >
                  <FormattedMessage defaultMessage="Delete" description="MCP server detail delete server action" />
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Root>
            <Button
              componentId="mlflow.mcp_registry.detail.create_version"
              type="primary"
              onClick={openCreateVersionModal}
            >
              <FormattedMessage
                defaultMessage="Create MCP server version"
                description="MCP server detail create version button"
              />
            </Button>
          </>
        }
      />
      <MCPServerTagsBox server={server} onTagsUpdated={refetchAll} />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <div css={{ flex: '0 0 320px', display: 'flex', flexDirection: 'column' }}>
          {versionsError ? (
            <Alert
              componentId="mlflow.mcp_registry.detail.versions_error"
              type="error"
              message={versionsError.message}
              closable={false}
            />
          ) : (
            <MCPServerVersionList
              versions={versions}
              selectedVersion={selectedVersion}
              onSelectVersion={setSelectedVersion}
              isLoading={versionsLoading}
              serverName={serverName}
              serverDisplayName={displayName}
              aliasesByVersion={aliasesByVersion}
              showEditAliasesModal={showEditAliasesModal}
            />
          )}
        </div>
        <div
          css={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            minWidth: 0,
            borderLeft: `1px solid ${theme.colors.border}`,
            overflow: 'hidden',
          }}
        >
          <MCPServerVersionDetail
            server={server}
            version={currentVersion}
            aliasesByVersion={aliasesByVersion}
            showEditAliasesModal={showEditAliasesModal}
            onEditMetadata={showEditMetadataModal}
          />
        </div>
      </div>
      {EditAliasesModal}
      {EditMCPServerVersionMetadataModal}
      {CreateMCPServerVersionModal}
      <UpdateVersionDisplayNameModal
        visible={editServerDisplayNameVisible}
        currentDisplayName={server.display_name || ''}
        isLoading={updateDisplayNameMutation.isLoading}
        error={updateDisplayNameMutation.error}
        onUpdate={(newDisplayName) => {
          updateDisplayNameMutation.mutate(newDisplayName || null, {
            onSuccess: () => setEditServerDisplayNameVisible(false),
          });
        }}
        onCancel={() => {
          updateDisplayNameMutation.reset();
          setEditServerDisplayNameVisible(false);
        }}
      />
      <ConfirmationModal
        componentId="mlflow.mcp_registry.detail.delete_server_modal"
        title={intl.formatMessage({
          defaultMessage: 'Delete MCP server',
          description: 'MCP server delete confirmation modal title',
        })}
        visible={deleteServerModalVisible}
        message={
          <FormattedMessage
            defaultMessage="Are you sure you want to delete this MCP server and all its versions? This action cannot be undone."
            description="MCP server delete confirmation message"
          />
        }
        isLoading={deleteServerMutation.isLoading}
        error={deleteServerMutation.error?.message ?? null}
        onConfirm={() => {
          deleteServerMutation.mutate(serverName, {
            onSuccess: () => {
              setDeleteServerModalVisible(false);
              navigate(MCPRegistryRoutes.mcpRegistryPageRoute);
            },
          });
        }}
        onCancel={() => {
          deleteServerMutation.reset();
          setDeleteServerModalVisible(false);
        }}
      />
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.MCP_REGISTRY, MCPServerDetailPage);

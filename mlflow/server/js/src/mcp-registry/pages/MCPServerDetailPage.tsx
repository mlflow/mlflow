import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  DropdownMenu,
  GenericSkeleton,
  Header,
  OverflowIcon,
  Spacer,
  TableSkeleton,
  Tag,
  Tooltip,
  Typography,
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
import {
  useMCPServerQuery,
  useMCPServerVersionsQuery,
  useLatestMCPServerVersionQuery,
  useMCPAccessBindingsQuery,
} from '../hooks/useMCPServerDetailQuery';
import { useDeleteAccessBindingMutation } from '../hooks/useAccessBindingMutation';
import { useUpdateMCPServerVersion } from '../hooks/useMCPServerVersionMutations';
import { AccessBindingModal } from '../components/AccessBindingModal';
import type { MCPAccessBinding } from '../types';
import { useCreateMCPServerVersionModal } from '../hooks/useCreateMCPServerVersionModal';
import { useUpdateMCPServerVersionMetadataModal } from '../hooks/useUpdateMCPServerVersionMetadataModal';
import { useDeleteServerModal } from '../hooks/useDeleteServerModal';
import { useEditDisplayNameModal } from '../hooks/useEditDisplayNameModal';
import { MCPServerVersionList } from '../components/MCPServerVersionList';
import { MCPServerVersionDetail } from '../components/MCPServerVersionDetail';
import { MCPServerTagsBox } from '../components/MCPServerTagsBox';
import { LATEST_ALIAS, resolveDisplayName } from '../utils';
import { useCurrentUserIsAdmin, useIsAuthAvailable } from '../../account/hooks';

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
  const isAuthAvailable = useIsAuthAvailable();
  const isUserAdmin = useCurrentUserIsAdmin();
  const canManage = !isAuthAvailable || isUserAdmin;
  const params = useParams<{ serverName: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const versionFromUrl = searchParams.get('version') ?? undefined;
  const serverName = decodeURIComponent(params.serverName ?? '');
  const [addBindingModalOpen, setAddBindingModalOpen] = useState(false);
  const [editBindingModalBinding, setEditBindingModalBinding] = useState<MCPAccessBinding | undefined>(undefined);
  const [deleteBindingModalBinding, setDeleteBindingModalBinding] = useState<MCPAccessBinding | undefined>(undefined);
  const deleteBindingMutation = useDeleteAccessBindingMutation();
  const updateVersionMutation = useUpdateMCPServerVersion(serverName);
  const { DeleteServerModal, openDeleteModal } = useDeleteServerModal({
    serverName,
    onDeleted: () => navigate(MCPRegistryRoutes.mcpRegistryPageRoute),
  });
  const { EditDisplayNameModal, openEditDisplayName } = useEditDisplayNameModal({ serverName });
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
    hasMoreVersions,
  } = useMCPServerVersionsQuery(serverName);
  const { data: latestVersion, refetch: refetchLatestVersion } = useLatestMCPServerVersionQuery(serverName);
  const { data: bindings, refetch: refetchBindings } = useMCPAccessBindingsQuery(serverName);

  const versionFoundInList = versionFromUrl && versions?.some((v) => v.version === versionFromUrl);

  const selectedVersion = useMemo(() => {
    if (!versions?.length) return undefined;
    if (versionFoundInList) return versionFromUrl;
    if (!versionFromUrl) return versions[0].version;
    return undefined;
  }, [versions, versionFromUrl, versionFoundInList]);

  const setSelectedVersion = useCallback(
    (version: string | undefined) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (version) {
            next.set('version', version);
          } else {
            next.delete('version');
          }
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  useEffect(() => {
    if (selectedVersion && selectedVersion !== versionFromUrl && !versionFromUrl) {
      setSelectedVersion(selectedVersion);
    }
  }, [selectedVersion, versionFromUrl, setSelectedVersion]);

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

  const versionBindings = useMemo(() => {
    if (!bindings || !selectedVersion) return bindings;
    const versionAliases = aliasesByVersion[selectedVersion] ?? [];
    return bindings.filter((b) => {
      if (b.server_version === selectedVersion) return true;
      if (b.resolved_version?.version === selectedVersion) return true;
      if (b.server_alias && versionAliases.includes(b.server_alias)) return true;
      return false;
    });
  }, [bindings, selectedVersion, aliasesByVersion]);

  const refetchAll = useCallback(async () => {
    await Promise.all([refetchServer(), refetchVersions(), refetchLatestVersion(), refetchBindings()]);
  }, [refetchServer, refetchVersions, refetchLatestVersion, refetchBindings]);

  const { CreateMCPServerVersionModal, openModal: openCreateVersionModal } = useCreateMCPServerVersionModal({
    serverName: serverName,
    latestVersion,
    onSuccess: ({ version }) => {
      setSelectedVersion(version);
    },
  });

  const { EditAliasesModal, showEditAliasesModal } = useEditAliasesModal({
    aliases: server?.aliases ?? [],
    getTitle: getAliasesModalTitle,
    onSave: async (version: string, existingAliases: string[], draftAliases: string[]) => {
      const add = draftAliases.filter((a) => !existingAliases.includes(a));
      const remove = existingAliases.filter((a) => !draftAliases.includes(a));
      await updateVersionMutation.mutateAsync({ version, aliases: { add, remove } });
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
        <Spacer shrinks={false} />
        <Header
          breadcrumbs={breadcrumbs}
          title={<GenericSkeleton css={{ height: theme.general.heightBase, width: 200 }} />}
          buttons={<GenericSkeleton css={{ height: theme.general.heightBase, width: 120 }} />}
        />
        <Spacer shrinks={false} />
        <div css={{ display: 'flex', gap: theme.spacing.lg }}>
          <div css={{ flex: '0 0 320px' }}>
            <TableSkeleton lines={6} />
          </div>
          <div css={{ flex: 1 }}>
            <TableSkeleton lines={4} />
          </div>
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
        title={
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            {displayName}
            {isAuthAvailable && isUserAdmin && (server.access_bindings?.length ?? 0) === 0 && (
              <Tooltip
                componentId="mlflow.mcp_registry.detail.unavailable_tooltip"
                content={
                  <FormattedMessage
                    defaultMessage="No access endpoints configured for this server"
                    description="Tooltip for unavailable label on MCP server detail page"
                  />
                }
              >
                <span css={{ cursor: 'default' }}>
                  <Tag componentId="mlflow.mcp_registry.detail.unavailable_tag" color="coral">
                    <FormattedMessage
                      defaultMessage="Unavailable"
                      description="Label for MCP server with no access endpoints"
                    />
                  </Tag>
                </span>
              </Tooltip>
            )}
          </span>
        }
        buttons={
          canManage ? (
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
                    onClick={() => openEditDisplayName(server.display_name || '')}
                  >
                    <FormattedMessage
                      defaultMessage="Edit display name"
                      description="MCP server detail edit server display name action"
                    />
                  </DropdownMenu.Item>
                  <DropdownMenu.Item componentId="mlflow.mcp_registry.detail.actions.delete" onClick={openDeleteModal}>
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
          ) : undefined
        }
      />
      {server.display_name && (
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs }}>
          {server.name}
        </Typography.Text>
      )}
      <MCPServerTagsBox server={server} />
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
              serverDisplayName={displayName}
              aliasesByVersion={aliasesByVersion}
              hasMoreVersions={hasMoreVersions}
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
            bindings={versionBindings}
            onAddBinding={() => setAddBindingModalOpen(true)}
            onEditBinding={(b) => setEditBindingModalBinding(b)}
            onDeleteBinding={(b) => setDeleteBindingModalBinding(b)}
          />
        </div>
      </div>
      {EditAliasesModal}
      {EditMCPServerVersionMetadataModal}
      {CreateMCPServerVersionModal}
      {EditDisplayNameModal}
      {DeleteServerModal}
      <AccessBindingModal
        visible={addBindingModalOpen || Boolean(editBindingModalBinding)}
        onCancel={() => {
          setAddBindingModalOpen(false);
          setEditBindingModalBinding(undefined);
        }}
        onSuccess={refetchAll}
        editBinding={editBindingModalBinding}
        lockedServer={serverName}
        scopedVersion={currentVersion?.version}
        scopedAliases={currentVersion ? aliasesByVersion[currentVersion.version] : undefined}
        createTitle={
          <FormattedMessage
            defaultMessage="Add access endpoint"
            description="MCP server add access endpoint modal title"
          />
        }
      />
      <ConfirmationModal
        componentId="mlflow.mcp_registry.detail.delete_binding_modal"
        title={intl.formatMessage({
          defaultMessage: 'Delete access endpoint',
          description: 'Access endpoint delete confirmation modal title',
        })}
        visible={Boolean(deleteBindingModalBinding)}
        message={
          <FormattedMessage
            defaultMessage="Are you sure you want to delete this access endpoint? This action cannot be undone."
            description="Access endpoint delete confirmation message"
          />
        }
        isLoading={deleteBindingMutation.isLoading}
        error={deleteBindingMutation.error?.message ?? null}
        onConfirm={() => {
          if (deleteBindingModalBinding) {
            deleteBindingMutation.mutate(
              { serverName, bindingId: deleteBindingModalBinding.binding_id },
              {
                onSuccess: () => {
                  setDeleteBindingModalBinding(undefined);
                  refetchAll();
                },
              },
            );
          }
        }}
        onCancel={() => {
          deleteBindingMutation.reset();
          setDeleteBindingModalBinding(undefined);
        }}
      />
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.MCP_REGISTRY, MCPServerDetailPage);

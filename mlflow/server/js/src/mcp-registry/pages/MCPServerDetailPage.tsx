import { useCallback, useEffect, useMemo } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  ColumnsIcon,
  DropdownMenu,
  GenericSkeleton,
  Header,
  OverflowIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  Spacer,
  TableSkeleton,
  Tag,
  Tooltip,
  Typography,
  ZoomMarqueeSelection,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { Link, useNavigate, useParams, useSearchParams } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useEditAliasesModal } from '../../common/hooks/useEditAliasesModal';
import MCPRegistryRoutes from '../routes';
import {
  useMCPServerQuery,
  useMCPServerVersionsQuery,
  useLatestMCPServerVersionQuery,
  useMCPAccessBindingsQuery,
} from '../hooks/useMCPServerDetailQuery';
import { useUpdateMCPServerVersion } from '../hooks/useMCPServerVersionMutations';
import { useCreateMCPServerVersionModal } from '../hooks/useCreateMCPServerVersionModal';
import { useUpdateMCPServerVersionMetadataModal } from '../hooks/useUpdateMCPServerVersionMetadataModal';
import { useDeleteServerModal } from '../hooks/useDeleteServerModal';
import { useEditDisplayNameModal } from '../hooks/useEditDisplayNameModal';
import { MCPServerVersionList } from '../components/MCPServerVersionList';
import { MCPServerVersionDetail } from '../components/MCPServerVersionDetail';
import { MCPServerVersionCompare } from '../components/MCPServerVersionCompare';
import { MCPServerTagsBox } from '../components/MCPServerTagsBox';
import { useMCPServerDetailViewState, MCPServerDetailViewMode } from '../hooks/useMCPServerDetailViewState';
import { LATEST_ALIAS, resolveDisplayName } from '../utils';
import { useServerState } from '../hooks/useServerState';

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
  const [searchParams, setSearchParams] = useSearchParams();
  const versionFromUrl = searchParams.get('version') ?? undefined;
  const serverName = decodeURIComponent(params.serverName ?? '');
  const updateVersionMutation = useUpdateMCPServerVersion(serverName);
  const { DeleteServerModal, openDeleteModal } = useDeleteServerModal({
    serverName,
    onDeleted: () => navigate(MCPRegistryRoutes.mcpRegistryPageRoute),
  });
  const { EditDisplayNameModal, openEditDisplayName } = useEditDisplayNameModal({ serverName });
  const { data: server, isLoading: serverLoading, error: serverError } = useMCPServerQuery(serverName);
  const {
    data: versions,
    isLoading: versionsLoading,
    error: versionsError,
    hasMoreVersions,
  } = useMCPServerVersionsQuery(serverName);
  const { data: latestVersion } = useLatestMCPServerVersionQuery(serverName);
  const { data: bindings } = useMCPAccessBindingsQuery(serverName);

  const { canUpdate, canDelete, isDimmed, isUnavailable } = useServerState(server);

  const {
    viewState,
    selectedVersion: hookSelectedVersion,
    setSelectedVersion: hookSetSelectedVersion,
    setPreviewMode,
    setCompareMode,
    setComparedVersion,
    switchSides,
  } = useMCPServerDetailViewState(versions);

  const versionFoundInList = versionFromUrl && versions?.some((v) => v.version === versionFromUrl);

  useEffect(() => {
    if (!versions?.length) {
      hookSetSelectedVersion(undefined);
      return;
    }
    const currentStillValid = versions.some((v) => v.version === hookSelectedVersion);
    if (!currentStillValid) {
      const urlVersion = versionFoundInList ? versionFromUrl : undefined;
      hookSetSelectedVersion(urlVersion ?? versions[0].version);
    }
    if (viewState.mode === MCPServerDetailViewMode.COMPARE && versions.length < 2) {
      setPreviewMode();
    } else if (viewState.comparedVersion && !versions.some((v) => v.version === viewState.comparedVersion)) {
      setComparedVersion(
        versions[0]?.version === hookSelectedVersion
          ? (versions[1]?.version ?? '')
          : (versions[0]?.version ?? ''),
      );
    }
  }, [
    versions,
    hookSelectedVersion,
    versionFromUrl,
    versionFoundInList,
    viewState.comparedVersion,
    viewState.mode,
    setComparedVersion,
    hookSetSelectedVersion,
    setPreviewMode,
  ]);

  const selectedVersion = hookSelectedVersion;

  const setSelectedVersion = useCallback(
    (version: string | undefined) => {
      hookSetSelectedVersion(version);
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
    [hookSetSelectedVersion, setSearchParams],
  );

  const currentVersion = versions?.find((v) => v.version === selectedVersion);

  const comparedVersionEntity = useMemo(
    () => versions?.find((v) => v.version === viewState.comparedVersion),
    [versions, viewState.comparedVersion],
  );

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

  const filterBindingsForVersion = useCallback(
    (version?: string) => {
      if (!bindings || !version) return undefined;
      const versionAliases = aliasesByVersion[version] ?? [];
      return bindings.filter((b) => {
        if (b.server_version === version) return true;
        if (b.resolved_version?.version === version) return true;
        if (b.server_alias && versionAliases.includes(b.server_alias)) return true;
        return false;
      });
    },
    [bindings, aliasesByVersion],
  );

  const versionBindings = useMemo(
    () => filterBindingsForVersion(selectedVersion) ?? bindings,
    [filterBindingsForVersion, selectedVersion, bindings],
  );

  const comparedBindings = useMemo(
    () => filterBindingsForVersion(viewState.comparedVersion),
    [filterBindingsForVersion, viewState.comparedVersion],
  );

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
            {isDimmed && (
              <Tooltip
                componentId="mlflow.mcp_registry.detail.unavailable_tooltip"
                content={
                  isUnavailable ? (
                    <FormattedMessage
                      defaultMessage="No access endpoints configured for this server"
                      description="Tooltip for unavailable label on MCP server detail page"
                    />
                  ) : (
                    <FormattedMessage
                      defaultMessage="Server status is not active"
                      description="Tooltip for inactive status label on MCP server detail page"
                    />
                  )
                }
              >
                <span css={{ cursor: 'default' }}>
                  <Tag componentId="mlflow.mcp_registry.detail.unavailable_tag" color="coral">
                    {isUnavailable ? (
                      <FormattedMessage
                        defaultMessage="Unavailable"
                        description="Label for MCP server with no access endpoints"
                      />
                    ) : (
                      server.status
                    )}
                  </Tag>
                </span>
              </Tooltip>
            )}
          </span>
        }
        buttons={
          canUpdate || canDelete ? (
            <>
              {(canUpdate || canDelete) && (
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
                    {canUpdate && (
                      <DropdownMenu.Item
                        componentId="mlflow.mcp_registry.detail.actions.edit_display_name"
                        onClick={() => openEditDisplayName(server.display_name || '')}
                      >
                        <FormattedMessage
                          defaultMessage="Edit display name"
                          description="MCP server detail edit server display name action"
                        />
                      </DropdownMenu.Item>
                    )}
                    {canDelete && (
                      <DropdownMenu.Item
                        componentId="mlflow.mcp_registry.detail.actions.delete"
                        onClick={openDeleteModal}
                      >
                        <FormattedMessage
                          defaultMessage="Delete"
                          description="MCP server detail delete server action"
                        />
                      </DropdownMenu.Item>
                    )}
                  </DropdownMenu.Content>
                </DropdownMenu.Root>
              )}
              {canUpdate && (
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
              )}
            </>
          ) : undefined
        }
      />
      {server.display_name && (
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs }}>
          {server.name}
        </Typography.Text>
      )}
      {server.description && (
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs }}>
          {server.description}
        </Typography.Text>
      )}
      <MCPServerTagsBox server={server} />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <div css={{ flex: '0 0 320px', display: 'flex', flexDirection: 'column' }}>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <SegmentedControlGroup
              name="mcp-server-detail-view"
              value={viewState.mode}
              componentId="mlflow.mcp_registry.detail.view_toggle"
            >
              <SegmentedControlButton value={MCPServerDetailViewMode.PREVIEW} onClick={setPreviewMode}>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <ZoomMarqueeSelection />
                  <FormattedMessage defaultMessage="Preview" description="MCP server detail preview tab" />
                </div>
              </SegmentedControlButton>
              <SegmentedControlButton
                value={MCPServerDetailViewMode.COMPARE}
                onClick={setCompareMode}
                disabled={!versions?.length || versions.length < 2}
              >
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <ColumnsIcon />
                  <FormattedMessage defaultMessage="Compare" description="MCP server detail compare tab" />
                </div>
              </SegmentedControlButton>
            </SegmentedControlGroup>
          </div>
          <Spacer shrinks={false} size="sm" />
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
              comparedVersion={viewState.comparedVersion}
              mode={viewState.mode}
              onSelectVersion={setSelectedVersion}
              onSelectComparedVersion={setComparedVersion}
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
          {viewState.mode === MCPServerDetailViewMode.COMPARE ? (
            <MCPServerVersionCompare
              baselineVersion={currentVersion}
              comparedVersion={comparedVersionEntity}
              aliasesByVersion={aliasesByVersion}
              baselineBindings={versionBindings}
              comparedBindings={comparedBindings}
              onSwitchSides={switchSides}
            />
          ) : (
            <MCPServerVersionDetail
              server={server}
              version={currentVersion}
              aliasesByVersion={aliasesByVersion}
              showEditAliasesModal={showEditAliasesModal}
              onEditMetadata={canUpdate ? showEditMetadataModal : undefined}
              bindings={versionBindings}
            />
          )}
        </div>
      </div>
      {EditAliasesModal}
      {EditMCPServerVersionMetadataModal}
      {CreateMCPServerVersionModal}
      {EditDisplayNameModal}
      {DeleteServerModal}
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.MCP_REGISTRY, MCPServerDetailPage);

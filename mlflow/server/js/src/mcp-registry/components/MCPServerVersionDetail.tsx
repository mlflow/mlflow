import { useEffect, useMemo, useState } from 'react';
import { tagListStyles, noShrinkStyles, flexColumnGapStyles } from '../styles';
import {
  Alert,
  Button,
  PencilIcon,
  CodeIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Spacer,
  Tabs,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPAccessEndpoint, MCPServer, MCPServerVersion } from '../types';
import { MCPStatus } from '../types';
import { STATUS_TAG_COLOR, STATUS_TRANSITIONS, sanitizeHref } from '../utils';
import { useServerState } from '../hooks/useServerState';
import { deriveClientName } from '../installInstructions';
import { AccessEndpointsSubsection } from './AccessEndpointsSubsection';
import { ServerJSONSection, ToolsSection } from './ServerJSONSection';
import { useAddAccessEndpointModal } from '../hooks/useAddAccessEndpointModal';
import { useEditAccessEndpointModal } from '../hooks/useEditAccessEndpointModal';
import { useDeleteAccessEndpointModal } from '../hooks/useDeleteAccessEndpointModal';
import { MCPServerAliasesCell } from './MCPServerAliasesCell';
import { KeyValueTag } from '../../common/components/KeyValueTag';
import { AddToolsModal } from './AddToolsModal';
import { useDeleteVersionModal } from '../hooks/useDeleteVersionModal';
import { useUpdateMCPServerVersion } from '../hooks/useMCPServerVersionMutations';
import Utils from '../../common/utils/Utils';

const STATUS_OPTIONS = [MCPStatus.DRAFT, MCPStatus.ACTIVE, MCPStatus.DEPRECATED, MCPStatus.DELETED];

export const MCPServerVersionDetail = ({
  server,
  version,
  aliasesByVersion,
  showEditAliasesModal,
  onEditMetadata,
  endpoints,
}: {
  server: MCPServer;
  version?: MCPServerVersion;
  aliasesByVersion: Record<string, string[]>;
  showEditAliasesModal?: (versionNumber: string) => void;
  onEditMetadata?: (version: MCPServerVersion) => void;
  endpoints?: MCPAccessEndpoint[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { canUpdate, canDelete } = useServerState(server);

  const [addToolsModalVisible, setAddToolsModalVisible] = useState(false);
  const [editingStatusVersion, setEditingStatusVersion] = useState<string>();
  const [pendingStatus, setPendingStatus] = useState<{ version: string; status: MCPStatus }>();
  const updateVersionMutation = useUpdateMCPServerVersion(server.name);
  const hasRemotes = (version?.server_json?.remotes ?? []).length > 0;
  const derivedName = useMemo(() => deriveClientName(server.name), [server.name]);
  const { DeleteVersionModal, openDeleteVersionModal } = useDeleteVersionModal({ serverName: server.name });
  const { AddAccessEndpointModal, openAddEndpoint } = useAddAccessEndpointModal({
    serverName: server.name,
    scopedVersion: version?.version,
    scopedAliases: version ? aliasesByVersion[version.version] : undefined,
  });
  const { EditAccessEndpointModal, openEditEndpoint } = useEditAccessEndpointModal({
    serverName: server.name,
    scopedVersion: version?.version,
    scopedAliases: version ? aliasesByVersion[version.version] : undefined,
  });
  const { DeleteAccessEndpointModal, openDeleteEndpoint } = useDeleteAccessEndpointModal({ serverName: server.name });
  const isEditingStatus = editingStatusVersion === version?.version;
  const versionNumber = version?.version;
  const versionStatus = version?.status;

  useEffect(() => {
    setEditingStatusVersion(undefined);
    setPendingStatus(undefined);
    updateVersionMutation.reset();
  }, [versionNumber]); // eslint-disable-line react-hooks/exhaustive-deps -- reset() creates new ref

  useEffect(() => {
    setPendingStatus((current) => {
      if (current && current.version === versionNumber && current.status === versionStatus) {
        return undefined;
      }
      return current;
    });
  }, [versionNumber, versionStatus]);

  if (!version) {
    return (
      <div
        css={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.lg,
        }}
      >
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Select a version to view details."
            description="MCP server detail placeholder when no version is selected"
          />
        </Typography.Text>
      </div>
    );
  }

  const displayedStatus = pendingStatus?.version === version.version ? pendingStatus.status : version.status;
  const statusSelectLabel = intl.formatMessage({
    defaultMessage: 'Version status',
    description: 'Aria label for MCP server version status selector',
  });
  const handleStatusChange = (nextStatus: MCPStatus) => {
    setEditingStatusVersion(undefined);
    if (nextStatus === displayedStatus) {
      return;
    }

    const clearPendingStatus = () => {
      setPendingStatus((current) =>
        current?.version === version.version && current.status === nextStatus ? undefined : current,
      );
    };

    setPendingStatus({ version: version.version, status: nextStatus });
    updateVersionMutation.mutate(
      { version: version.version, status: nextStatus },
      {
        onError: clearPendingStatus,
      },
    );
  };

  return (
    <div
      css={{
        flex: 1,
        paddingTop: theme.spacing.md,
        paddingRight: 0,
        paddingBottom: theme.spacing.md,
        paddingLeft: theme.spacing.md,
        overflow: 'auto',
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: theme.spacing.sm }}>
        <div css={{ minWidth: 0, flex: 1 }}>
          <Typography.Title level={3} withoutMargins>
            <FormattedMessage
              defaultMessage="Viewing version {version}"
              description="MCP server version detail heading"
              values={{ version: version.version }}
            />
          </Typography.Title>
          {version.server_json?.description && (
            <Typography.Hint css={{ marginTop: theme.spacing.xs }}>{version.server_json.description}</Typography.Hint>
          )}
        </div>
        {canDelete && (
          <div css={{ display: 'flex', gap: theme.spacing.sm, ...noShrinkStyles }}>
            <Button
              componentId="mlflow.mcp_registry.detail.delete_version"
              icon={<TrashIcon />}
              type="primary"
              danger
              onClick={() => openDeleteVersionModal(version.version)}
            >
              <FormattedMessage defaultMessage="Delete version" description="MCP server delete version button" />
            </Button>
          </div>
        )}
      </div>

      <Spacer shrinks={false} />
      {updateVersionMutation.error && (
        <Alert
          componentId="mlflow.mcp_registry.detail.version.status_update_error"
          type="error"
          closable
          onClose={() => updateVersionMutation.reset()}
          message={
            updateVersionMutation.error instanceof Error
              ? updateVersionMutation.error.message
              : String(updateVersionMutation.error)
          }
          css={{ marginBottom: theme.spacing.sm }}
        />
      )}
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: '120px 1fr',
          gridAutoRows: `minmax(${theme.typography.lineHeightLg}, auto)`,
          alignItems: 'flex-start',
          rowGap: theme.spacing.xs,
          columnGap: theme.spacing.sm,
        }}
      >
        <Typography.Text bold>
          <FormattedMessage defaultMessage="Name:" description="MCP server version detail name label" />
        </Typography.Text>
        <Typography.Text>{server.name}</Typography.Text>

        <Typography.Text bold>
          <FormattedMessage defaultMessage="Aliases:" description="MCP server version detail aliases label" />
        </Typography.Text>
        <div>
          {(aliasesByVersion[version.version] ?? []).length > 0 || canUpdate ? (
            <MCPServerAliasesCell
              aliases={aliasesByVersion[version.version] ?? []}
              onEdit={canUpdate ? () => showEditAliasesModal?.(version.version) : undefined}
            />
          ) : (
            <Typography.Hint>—</Typography.Hint>
          )}
        </div>

        <Typography.Text bold>
          <FormattedMessage defaultMessage="Status:" description="MCP server version detail status label" />
        </Typography.Text>
        <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          {isEditingStatus ? (
            <DialogCombobox
              id="mcp-registry-version-status"
              componentId="mlflow.mcp_registry.detail.version.status_select"
              label={statusSelectLabel}
              value={[displayedStatus]}
              open
            >
              <DialogComboboxTrigger
                aria-label={statusSelectLabel}
                withInlineLabel={false}
                renderDisplayedValue={(status) => formatStatusLabel(status as MCPStatus)}
                disabled={updateVersionMutation.isLoading}
                allowClear={false}
                width={160}
              />
              <DialogComboboxContent
                matchTriggerWidth
                onEscapeKeyDown={() => setEditingStatusVersion(undefined)}
                onPointerDownOutside={() => setEditingStatusVersion(undefined)}
              >
                <DialogComboboxOptionList>
                  {STATUS_OPTIONS.map((status) => (
                    <DialogComboboxOptionListSelectItem
                      key={status}
                      value={status}
                      checked={status === displayedStatus}
                      disabled={status !== displayedStatus && !STATUS_TRANSITIONS[displayedStatus]?.includes(status)}
                      onChange={(nextStatus) => handleStatusChange(nextStatus as MCPStatus)}
                    >
                      {formatStatusLabel(status)}
                    </DialogComboboxOptionListSelectItem>
                  ))}
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
          ) : (
            <>
              <Tag componentId="mlflow.mcp_registry.detail.version_status" color={STATUS_TAG_COLOR[displayedStatus]}>
                {displayedStatus}
              </Tag>
              {canUpdate && (
                <Button
                  componentId="mlflow.mcp_registry.detail.version.edit_status"
                  size="small"
                  icon={<PencilIcon />}
                  aria-label={intl.formatMessage({
                    defaultMessage: 'Edit version status',
                    description: 'Aria label for edit MCP server version status button',
                  })}
                  disabled={updateVersionMutation.isLoading}
                  onClick={() => {
                    setEditingStatusVersion(version.version);
                  }}
                />
              )}
            </>
          )}
        </span>

        {sanitizeHref(version.server_json?.websiteUrl) && (
          <>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Website:" description="MCP server version detail website label" />
            </Typography.Text>
            <Typography.Link
              componentId="mlflow.mcp_registry.detail.website"
              href={sanitizeHref(version.server_json?.websiteUrl)}
              target="_blank"
              rel="noopener noreferrer"
            >
              {version.server_json?.websiteUrl}
            </Typography.Link>
          </>
        )}

        {sanitizeHref(version.server_json?.repository?.url) && (
          <>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Repository:" description="MCP server version detail repository label" />
            </Typography.Text>
            <Typography.Link
              componentId="mlflow.mcp_registry.detail.repository"
              href={sanitizeHref(version.server_json?.repository?.url)}
              target="_blank"
              rel="noopener noreferrer"
            >
              {version.server_json?.repository?.url}
            </Typography.Link>
          </>
        )}

        <Typography.Text bold>
          <FormattedMessage defaultMessage="Created:" description="MCP server version detail registered at label" />
        </Typography.Text>
        <Typography.Text>
          {version.creation_timestamp ? Utils.formatTimestamp(version.creation_timestamp, intl) : '—'}
        </Typography.Text>

        <Typography.Text bold>
          <FormattedMessage defaultMessage="Metadata:" description="MCP server version detail metadata label" />
        </Typography.Text>
        <div>
          <div css={tagListStyles(theme)}>
            {Object.keys(version.tags ?? {}).length > 0
              ? Object.entries(version.tags ?? {}).map(([key, value]) => (
                  <KeyValueTag css={{ margin: 0 }} key={key} tag={{ key, value }} />
                ))
              : !onEditMetadata && <Typography.Hint>—</Typography.Hint>}
            {onEditMetadata &&
              (Object.keys(version.tags ?? {}).length > 0 ? (
                <Button
                  componentId="mlflow.mcp_registry.detail.version.edit_metadata"
                  size="small"
                  icon={<PencilIcon />}
                  aria-label={intl.formatMessage({
                    defaultMessage: 'Edit metadata',
                    description: 'Aria label for edit metadata button',
                  })}
                  onClick={() => onEditMetadata(version)}
                />
              ) : (
                <Button
                  componentId="mlflow.mcp_registry.detail.version.add_metadata"
                  size="small"
                  type="link"
                  onClick={() => onEditMetadata(version)}
                >
                  <FormattedMessage defaultMessage="Add" description="MCP server version detail add metadata button" />
                </Button>
              ))}
          </div>
        </div>
      </div>

      <Tabs.Root
        key={version.version}
        componentId="mlflow.mcp_registry.detail.version_tabs"
        valueHasNoPii
        defaultValue="connect"
        css={{ marginTop: theme.spacing.md, '& svg': { width: 14, height: 14 } }}
      >
        <Tabs.List>
          <Tabs.Trigger value="connect">
            <FormattedMessage defaultMessage="Connect" description="MCP server version detail connect tab" />
          </Tabs.Trigger>
          <Tabs.Trigger value="tools">
            <FormattedMessage defaultMessage="Tools" description="MCP server version detail tools tab" />
          </Tabs.Trigger>
        </Tabs.List>

        <Tabs.Content value="connect" css={{ ...flexColumnGapStyles(theme, theme.spacing.md) }}>
          <AccessEndpointsSubsection
            endpoints={endpoints ?? []}
            derivedName={derivedName}
            server={server}
            onAddEndpoint={openAddEndpoint}
            onEditEndpoint={openEditEndpoint}
            onDeleteEndpoint={openDeleteEndpoint}
          />
          <ServerJSONSection serverJson={version.server_json} server={server} version={version} />
        </Tabs.Content>

        <Tabs.Content value="tools" css={{ paddingTop: theme.spacing.md }}>
          {hasRemotes && canUpdate && (
            <div css={{ marginBottom: theme.spacing.md }}>
              <Button
                componentId="mlflow.mcp_registry.detail.add_tools"
                icon={<CodeIcon />}
                onClick={() => setAddToolsModalVisible(true)}
              >
                <FormattedMessage
                  defaultMessage="Auto-discover tools"
                  description="MCP server auto-discover tools button"
                />
              </Button>
            </div>
          )}
          {version.tools && version.tools.length > 0 ? (
            <ToolsSection tools={version.tools} />
          ) : (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="No tools registered for this version."
                description="MCP server tools tab empty state"
              />
            </Typography.Text>
          )}
        </Tabs.Content>
      </Tabs.Root>

      <AddToolsModal
        visible={addToolsModalVisible}
        serverName={server.name}
        version={version.version}
        onClose={() => setAddToolsModalVisible(false)}
      />
      {DeleteVersionModal}
      {AddAccessEndpointModal}
      {EditAccessEndpointModal}
      {DeleteAccessEndpointModal}
    </div>
  );
};

const formatStatusLabel = (status: MCPStatus) => status.charAt(0).toUpperCase() + status.slice(1);

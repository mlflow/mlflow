import { useEffect, useRef, useState } from 'react';
import { tagListStyles, textEllipsisStyles, mcpIconStyles, noShrinkStyles } from '../styles';
import {
  Button,
  McpIcon,
  PencilIcon,
  Spacer,
  Tabs,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPServer, MCPServerVersion } from '../types';
import { STATUS_TAG_COLOR, resolveDisplayName, sanitizeHref } from '../utils';
import { ServerJSONSection, ToolsSection } from './ServerJSONSection';
import { ConfirmationModal } from '../../admin/ConfirmationModal';
import { MCPServerAliasesCell } from './MCPServerAliasesCell';
import { useUpdateMCPServerVersion, useDeleteMCPServerVersion } from '../hooks/useMCPServerVersionMutations';
import { KeyValueTag } from '../../common/components/KeyValueTag';
import { EditVersionModal } from './EditVersionModal';
import { useCurrentUserIsAdmin, useIsAuthAvailable } from '../../account/hooks';
import Utils from '../../common/utils/Utils';

export const MCPServerVersionDetail = ({
  server,
  version,
  aliasesByVersion,
  showEditAliasesModal,
  onEditMetadata,
}: {
  server: MCPServer;
  version?: MCPServerVersion;
  aliasesByVersion: Record<string, string[]>;
  showEditAliasesModal?: (versionNumber: string) => void;
  onEditMetadata?: (version: MCPServerVersion) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isAuthAvailable = useIsAuthAvailable();
  const isUserAdmin = useCurrentUserIsAdmin();
  const isAdmin = !isAuthAvailable || isUserAdmin;
  const [editVersionModalVisible, setEditVersionModalVisible] = useState(false);
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [localHiddenOptions, setLocalHiddenOptions] = useState<string[] | undefined>(undefined);
  const currentVersionRef = useRef(version?.version);
  currentVersionRef.current = version?.version;

  useEffect(() => {
    setLocalHiddenOptions(undefined);
  }, [version?.version]);
  const updateVersionMutation = useUpdateMCPServerVersion(server.name);
  const deleteVersionMutation = useDeleteMCPServerVersion(server.name);

  const handleToggleConnectOption = (key: string, visible: boolean) => {
    if (!version) return;
    const toggledVersion = version.version;
    const current = localHiddenOptions ?? version.hidden_connect_options ?? [];
    const updated = visible ? current.filter((k) => k !== key) : [...current.filter((k) => k !== key), key];
    setLocalHiddenOptions(updated);
    updateVersionMutation.mutate(
      {
        version: toggledVersion,
        hiddenConnectOptions: updated.length > 0 ? updated : null,
      },
      {
        onError: () => {
          if (currentVersionRef.current === toggledVersion) {
            setLocalHiddenOptions(current);
          }
        },
      },
    );
  };

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

  const displayName = resolveDisplayName(server);
  const versionDisplayName = version.display_name || version.server_json?.title;
  const showVersionDisplayName = versionDisplayName && versionDisplayName !== displayName;

  return (
    <div css={{ flex: 1, padding: theme.spacing.md, overflow: 'auto' }}>
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: theme.spacing.sm }}>
        <div css={{ minWidth: 0, flex: 1 }}>
          <Typography.Title level={3} withoutMargins>
            <FormattedMessage
              defaultMessage="Viewing version {version}"
              description="MCP server version detail heading"
              values={{ version: version.version }}
            />
          </Typography.Title>
          {showVersionDisplayName && (
            <Typography.Text color="secondary" css={textEllipsisStyles} title={versionDisplayName}>
              {versionDisplayName}
            </Typography.Text>
          )}
          {version.server_json?.description && (
            <Typography.Hint css={{ marginTop: theme.spacing.xs }}>{version.server_json.description}</Typography.Hint>
          )}
        </div>
        {isAdmin && (
          <div css={{ display: 'flex', gap: theme.spacing.sm, ...noShrinkStyles }}>
            <Button
              componentId="mlflow.mcp_registry.detail.edit_version"
              icon={<PencilIcon />}
              onClick={() => setEditVersionModalVisible(true)}
            >
              <FormattedMessage defaultMessage="Edit" description="MCP server edit version button" />
            </Button>
            <Button
              componentId="mlflow.mcp_registry.detail.delete_version"
              icon={<TrashIcon />}
              type="primary"
              danger
              onClick={() => setDeleteModalVisible(true)}
            >
              <FormattedMessage defaultMessage="Delete version" description="MCP server delete version button" />
            </Button>
          </div>
        )}
      </div>

      <Spacer shrinks={false} />
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <McpIcon css={mcpIconStyles(theme)} />
        <div css={{ display: 'flex', flexDirection: 'column' }}>
          <Typography.Text bold>{displayName}</Typography.Text>
          <Typography.Text color="secondary" size="sm">
            {server.name}
          </Typography.Text>
        </div>
      </div>

      <Spacer shrinks={false} />
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
          {(aliasesByVersion[version.version] ?? []).length > 0 || isAdmin ? (
            <MCPServerAliasesCell
              aliases={aliasesByVersion[version.version] ?? []}
              onEdit={isAdmin ? () => showEditAliasesModal?.(version.version) : undefined}
            />
          ) : (
            <Typography.Hint>—</Typography.Hint>
          )}
        </div>

        <Typography.Text bold>
          <FormattedMessage defaultMessage="Status:" description="MCP server version detail status label" />
        </Typography.Text>
        <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Tag componentId="mlflow.mcp_registry.detail.version_status" color={STATUS_TAG_COLOR[version.status]}>
            {version.status}
          </Tag>
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
          {version.tools && version.tools.length > 0 && (
            <Tabs.Trigger value="tools">
              <FormattedMessage defaultMessage="Tools" description="MCP server version detail tools tab" />
            </Tabs.Trigger>
          )}
        </Tabs.List>

        <Tabs.Content value="connect" css={{ paddingTop: theme.spacing.md }}>
          {version.server_json && (
            <ServerJSONSection
              serverJson={version.server_json}
              serverName={server.name}
              isAdmin={isAdmin}
              isAuthAvailable={isAuthAvailable}
              hiddenConnectOptions={localHiddenOptions ?? version.hidden_connect_options ?? undefined}
              onToggleConnectOption={handleToggleConnectOption}
            />
          )}
        </Tabs.Content>

        {version.tools && version.tools.length > 0 && (
          <Tabs.Content value="tools" css={{ paddingTop: theme.spacing.md }}>
            <ToolsSection tools={version.tools} />
          </Tabs.Content>
        )}
      </Tabs.Root>

      <EditVersionModal
        visible={editVersionModalVisible}
        server={server}
        version={version}
        aliasesByVersion={aliasesByVersion}
        onClose={() => setEditVersionModalVisible(false)}
      />

      <ConfirmationModal
        componentId="mlflow.mcp_registry.detail.delete_version_modal"
        title={intl.formatMessage({
          defaultMessage: 'Delete version',
          description: 'MCP server delete version confirmation modal title',
        })}
        visible={deleteModalVisible}
        message={
          <FormattedMessage
            defaultMessage="Are you sure you want to delete version {version}? This action cannot be undone."
            description="MCP server delete version confirmation message"
            values={{ version: version.version }}
          />
        }
        isLoading={deleteVersionMutation.isLoading}
        error={deleteVersionMutation.error?.message ?? null}
        onConfirm={() => {
          deleteVersionMutation.mutate(version.version, {
            onSuccess: () => setDeleteModalVisible(false),
          });
        }}
        onCancel={() => {
          deleteVersionMutation.reset();
          setDeleteModalVisible(false);
        }}
      />
    </div>
  );
};

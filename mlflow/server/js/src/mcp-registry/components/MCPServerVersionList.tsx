import { useCallback } from 'react';
import { Empty, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { RegistryVersionList } from '../../common/components/RegistryVersionList';
import { flexColumnGapStyles, flexRowWrapStyles, textEllipsisStyles } from '../styles';
import type { MCPServerVersion } from '../types';
import { MCPServerDetailViewMode } from '../types';
import { STATUS_TAG_COLOR } from '../utils';
import { MCPServerVersionDiffSelectorButton } from './MCPServerVersionDiffSelectorButton';
import { MCPServerAliasesCell } from './MCPServerAliasesCell';
import Utils from '../../common/utils/Utils';

const getVersionKey = (version: MCPServerVersion) => version.version;

const MCPServerVersionSummary = ({
  version,
  serverDisplayName,
  aliases,
}: {
  version: MCPServerVersion;
  serverDisplayName: string;
  aliases: string[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const rawTitle = version.server_json?.title;
  const versionTitle = rawTitle && rawTitle !== serverDisplayName ? rawTitle : undefined;

  return (
    <div css={flexColumnGapStyles(theme)}>
      <div css={flexRowWrapStyles(theme)}>
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="{version}"
            description="MCP server version item label"
            values={{ version: version.version }}
          />
        </Typography.Text>
        <Tag componentId="mlflow.mcp_registry.detail.version_status_tag" color={STATUS_TAG_COLOR[version.status]}>
          {version.status}
        </Tag>
        <MCPServerAliasesCell aliases={aliases} />
      </div>
      {versionTitle && (
        <Typography.Text size="sm" color="secondary" css={textEllipsisStyles} title={versionTitle}>
          {versionTitle}
        </Typography.Text>
      )}
      {version.creation_timestamp && (
        <Typography.Text size="sm" color="secondary">
          {Utils.formatTimestamp(version.creation_timestamp, intl)}
        </Typography.Text>
      )}
    </div>
  );
};

export const MCPServerVersionList = ({
  versions,
  selectedVersion,
  comparedVersion,
  mode = MCPServerDetailViewMode.PREVIEW,
  onSelectVersion,
  onSelectComparedVersion,
  isLoading,
  serverDisplayName,
  aliasesByVersion,
  hasMoreVersions,
}: {
  versions?: MCPServerVersion[];
  selectedVersion?: string;
  comparedVersion?: string;
  mode?: MCPServerDetailViewMode;
  onSelectVersion: (version: string) => void;
  onSelectComparedVersion?: (version: string) => void;
  isLoading?: boolean;
  serverDisplayName: string;
  aliasesByVersion: Record<string, string[]>;
  hasMoreVersions?: boolean;
}) => {
  const intl = useIntl();
  const renderVersion = useCallback(
    (version: MCPServerVersion) => (
      <MCPServerVersionSummary
        version={version}
        serverDisplayName={serverDisplayName}
        aliases={aliasesByVersion[version.version] || []}
      />
    ),
    [serverDisplayName, aliasesByVersion],
  );

  return (
    <RegistryVersionList
      versions={versions}
      getVersionKey={getVersionKey}
      renderVersion={renderVersion}
      selectedKey={selectedVersion}
      onSelect={onSelectVersion}
      header={intl.formatMessage({
        defaultMessage: 'Versions',
        description: 'Header for the version column in the MCP server versions table',
      })}
      componentId="mlflow.mcp_registry.detail.versions"
      emptyState={
        <Empty
          title={
            <FormattedMessage defaultMessage="No versions" description="Empty state when MCP server has no versions" />
          }
          description={null}
        />
      }
      isLoading={isLoading}
      hasMoreVersions={hasMoreVersions}
      compareMode={
        mode === MCPServerDetailViewMode.COMPARE
          ? {
              comparedKey: comparedVersion,
              renderControls: (version, { isSelected, isCompared }) => (
                <MCPServerVersionDiffSelectorButton
                  isSelectedBaseline={isSelected}
                  isSelectedCompared={isCompared}
                  onSelectBaseline={() => onSelectVersion(version)}
                  onSelectCompared={() => onSelectComparedVersion?.(version)}
                />
              ),
            }
          : undefined
      }
    />
  );
};

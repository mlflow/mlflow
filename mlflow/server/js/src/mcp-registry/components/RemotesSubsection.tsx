import { useState } from 'react';
import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  InfoIcon,
  Popover,
  Tag,
  Tooltip,
  Typography,
  VisibleIcon,
  VisibleOffIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ConnectionSource } from '../types';
import type { ServerJSONPayload } from '../types';
import {
  expandableRowButtonStyles,
  chevronContainerStyles,
  borderedSectionContainerStyles,
  borderedListContainerStyles,
  borderedListItemStyles,
  expandedContentPanelStyles,
  popoverTriggerStyles,
  monoEllipsisStyles,
} from '../styles';
import { ViewDetailsDrawer, DetailField } from './ViewDetailsDrawer';
import { ConnectionInstructions } from './ConnectionInstructions';

export const RemotesSubsection = ({
  remotes,
  serverName,
  derivedName,
  serverJson,
  isAdmin,
  isAuthAvailable,
  hiddenConnectOptions,
  onToggleConnectOption,
}: {
  remotes: NonNullable<ServerJSONPayload['remotes']>;
  serverName: string;
  derivedName: string;
  serverJson: ServerJSONPayload;
  isAdmin?: boolean;
  isAuthAvailable?: boolean;
  hiddenConnectOptions?: string[];
  onToggleConnectOption?: (key: string, visible: boolean) => void;
}) => {
  const showVisibilityControls = isAuthAvailable && isAdmin;
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  return (
    <div>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, marginBottom: theme.spacing.sm }}>
        <Typography.Text bold>
          <FormattedMessage defaultMessage="Official endpoints" description="MCP server official endpoints subsection heading" />
        </Typography.Text>
        <Popover.Root componentId="mlflow.mcp_registry.detail.remotes_help">
          <Popover.Trigger css={popoverTriggerStyles(theme)}>
            <InfoIcon />
          </Popover.Trigger>
          <Popover.Content align="start" css={{ maxWidth: 360 }}>
            <Typography.Paragraph withoutMargins>
              <FormattedMessage defaultMessage="Remote endpoints provided by the server maintainer for direct connections." description="Help text for MCP server official endpoints subsection" />
            </Typography.Paragraph>
            <Popover.Arrow />
          </Popover.Content>
        </Popover.Root>
      </div>
      <div css={borderedSectionContainerStyles(theme)}>
        {remotes.map((remote, index) => (
          <RemoteRow
            key={`${remote.type}-${remote.url ?? index}`}
            remote={remote}
            serverName={serverName}
            derivedName={derivedName}
            serverJson={serverJson}
            expanded={expandedIndex === index}
            onToggle={() => setExpandedIndex(expandedIndex === index ? null : index)}
            showTopBorder={index > 0}
            showVisibilityControls={showVisibilityControls}
            isHidden={hiddenConnectOptions?.includes(`remote:${remote.url ?? remote.type}`) ?? false}
            onToggleVisibility={(visible) => onToggleConnectOption?.(`remote:${remote.url ?? remote.type}`, visible)}
          />
        ))}
      </div>
    </div>
  );
};

const RemoteRow = ({
  remote,
  serverName,
  derivedName,
  serverJson,
  expanded,
  onToggle,
  showTopBorder,
  showVisibilityControls,
  isHidden,
  onToggleVisibility,
}: {
  remote: NonNullable<ServerJSONPayload['remotes']>[number];
  serverName: string;
  derivedName: string;
  serverJson: ServerJSONPayload;
  expanded: boolean;
  onToggle: () => void;
  showTopBorder: boolean;
  showVisibilityControls?: boolean;
  isHidden?: boolean;
  onToggleVisibility?: (visible: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isVisible = !isHidden;
  const isDisabled = !isVisible;

  if (isDisabled && !showVisibilityControls) return null;

  return (
    <div
      css={{
        borderTop: showTopBorder ? `1px solid ${theme.colors.border}` : 'none',
        opacity: isDisabled ? 0.5 : 1,
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        aria-label={intl.formatMessage(
          {
            defaultMessage: '{action} remote {url}',
            description: 'Aria label for expanding/collapsing a remote row',
          },
          { action: expanded ? 'Collapse' : 'Expand', url: remote.url ?? remote.type },
        )}
        css={expandableRowButtonStyles(theme)}
      >
        <div css={chevronContainerStyles(theme)}>
          {expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        </div>
        <Tag componentId="mlflow.mcp_registry.detail.remote_transport_tag" color="charcoal" css={{ flexShrink: 0 }}>
          {remote.type}
        </Tag>
        <Typography.Text
          color="secondary"
          css={{ ...monoEllipsisStyles(theme), fontFamily: 'inherit' }}
        >
          <strong>{remote.type}</strong>{remote.url ? ` ${remote.url}` : ''}
        </Typography.Text>
        {showVisibilityControls && isDisabled && (
          <Tag componentId="mlflow.mcp_registry.detail.remote.disabled_tag" color="charcoal" css={{ flexShrink: 0 }}>
            <FormattedMessage defaultMessage="Disabled" description="Label for disabled remote endpoint" />
          </Tag>
        )}
        {showVisibilityControls && (
          <div css={{ flexShrink: 0 }} onClick={(e) => e.stopPropagation()}>
            <Tooltip
              componentId="mlflow.mcp_registry.detail.remote.visibility_tooltip"
              content={
                isVisible ? (
                  <FormattedMessage defaultMessage="Visible to developers. Click to hide." description="Tooltip for visible endpoint toggle" />
                ) : (
                  <FormattedMessage defaultMessage="Hidden from developers. Click to show." description="Tooltip for hidden endpoint toggle" />
                )
              }
            >
              <Button
                componentId="mlflow.mcp_registry.detail.remote.visibility_row"
                type="tertiary"
                size="small"
                icon={isVisible ? <VisibleIcon /> : <VisibleOffIcon />}
                onClick={() => onToggleVisibility?.(!!isHidden)}
              />
            </Tooltip>
          </div>
        )}
      </button>

      {expanded && (
        <div css={expandedContentPanelStyles(theme)}>
          <ConnectionInstructions
            source={ConnectionSource.REMOTE}
            remote={remote}
            derivedName={derivedName}
            detailLink={
              <ViewDetailsDrawer title={remote.url ?? remote.type}>
                <DetailField label={intl.formatMessage({ defaultMessage: 'Transport type', description: 'Remote transport type label' })} value={remote.type} tagColor="default" />
                {remote.url && <DetailField label={intl.formatMessage({ defaultMessage: 'URL', description: 'Remote URL label' })} value={remote.url} mono link />}
                {remote.headers && remote.headers.length > 0 && (
                  <div>
                    <Typography.Text bold size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                      <FormattedMessage
                        defaultMessage="Headers ({count})"
                        description="MCP server remote headers heading with count"
                        values={{ count: remote.headers.length }}
                      />
                    </Typography.Text>
                    <div css={borderedListContainerStyles(theme)}>
                      {remote.headers.map((header, i) => (
                        <div key={header.name} css={borderedListItemStyles(theme, i > 0)}>
                          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                            <Typography.Text bold size="sm" css={{ fontFamily: 'monospace' }}>
                              {header.name}
                            </Typography.Text>
                            {header.isRequired && (
                              <Tag componentId="mlflow.mcp_registry.detail.header_required" color="lemon">
                                <FormattedMessage defaultMessage="required" description="Header required badge" />
                              </Tag>
                            )}
                            {header.isSecret && (
                              <Tag componentId="mlflow.mcp_registry.detail.header_secret" color="coral">
                                <FormattedMessage defaultMessage="secret" description="Header secret badge" />
                              </Tag>
                            )}
                          </div>
                          {header.description && (
                            <Typography.Text color="secondary" size="sm">
                              {header.description}
                            </Typography.Text>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {remote.variables && Object.keys(remote.variables).length > 0 && (
                  <div>
                    <Typography.Text bold size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                      <FormattedMessage defaultMessage="URL Variables" description="MCP server remote URL variables heading" />
                    </Typography.Text>
                    <div css={borderedListContainerStyles(theme)}>
                      {Object.entries(remote.variables).map(([name, variable], i) => {
                        return (
                          <div key={name} css={borderedListItemStyles(theme, i > 0)}>
                            <Typography.Text bold size="sm" css={{ fontFamily: 'monospace' }}>
                              {`{${name}}`}
                            </Typography.Text>
                            {variable.description && (
                              <Typography.Text color="secondary" size="sm" css={{ display: 'block' }}>
                                {variable.description}
                              </Typography.Text>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </ViewDetailsDrawer>
            }
          />
        </div>
      )}
    </div>
  );
};

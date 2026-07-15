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
  ellipsisStyles,
  sectionHeadingRowStyles,
  flexRowStyles,
  noShrinkStyles,
  blockLabelStyles,
  monoFontStyles,
} from '../styles';
import { buildRemoteConnectOptionKey } from '../utils';
import { ViewDetailsDrawer, DetailField } from './ViewDetailsDrawer';
import { ConnectionInstructions } from './ConnectionInstructions';

export const RemotesSubsection = ({
  remotes,
  derivedName,
  isAdmin,
  isAuthAvailable,
  connectOptions,
  onToggleConnectOption,
}: {
  remotes: NonNullable<ServerJSONPayload['remotes']>;
  derivedName: string;
  isAdmin?: boolean;
  isAuthAvailable?: boolean;
  connectOptions?: Record<string, { hidden?: boolean }>;
  onToggleConnectOption?: (key: string, visible: boolean) => void;
}) => {
  const showVisibilityControls = isAuthAvailable && isAdmin;
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  return (
    <div>
      <div css={sectionHeadingRowStyles(theme)}>
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Official endpoints"
            description="MCP server official endpoints subsection heading"
          />
        </Typography.Text>
        <Popover.Root componentId="mlflow.mcp_registry.detail.remotes_help">
          <Popover.Trigger
            css={popoverTriggerStyles(theme)}
            aria-label={intl.formatMessage({
              defaultMessage: 'About official endpoints',
              description: 'Aria label for remotes subsection help popover',
            })}
          >
            <InfoIcon />
          </Popover.Trigger>
          <Popover.Content align="start" css={{ maxWidth: 360 }}>
            <Typography.Paragraph withoutMargins>
              <FormattedMessage
                defaultMessage="Remote endpoints provided by the server maintainer for direct connections."
                description="Help text for MCP server official endpoints subsection"
              />
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
            derivedName={derivedName}
            expanded={expandedIndex === index}
            onToggle={() => setExpandedIndex(expandedIndex === index ? null : index)}
            showTopBorder={index > 0}
            showVisibilityControls={showVisibilityControls}
            isHidden={connectOptions?.[buildRemoteConnectOptionKey(remote)]?.hidden ?? false}
            onToggleVisibility={(visible) => onToggleConnectOption?.(buildRemoteConnectOptionKey(remote), visible)}
          />
        ))}
      </div>
    </div>
  );
};

const RemoteRow = ({
  remote,
  derivedName,
  expanded,
  onToggle,
  showTopBorder,
  showVisibilityControls,
  isHidden,
  onToggleVisibility,
}: {
  remote: NonNullable<ServerJSONPayload['remotes']>[number];
  derivedName: string;
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
  const hasAdvancedDetails =
    (remote.headers && remote.headers.length > 0) || (remote.variables && Object.keys(remote.variables).length > 0);

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
        <div css={chevronContainerStyles(theme)}>{expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}</div>
        <Tag componentId="mlflow.mcp_registry.detail.remote_transport_tag" color="charcoal" css={noShrinkStyles}>
          {remote.type}
        </Tag>
        <Typography.Text color="secondary" css={ellipsisStyles(theme)}>
          {remote.url ?? ''}
        </Typography.Text>
        {showVisibilityControls && isDisabled && (
          <Tag componentId="mlflow.mcp_registry.detail.remote.disabled_tag" color="charcoal" css={noShrinkStyles}>
            <FormattedMessage defaultMessage="Disabled" description="Label for disabled remote endpoint" />
          </Tag>
        )}
        {showVisibilityControls && (
          <div css={noShrinkStyles} onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
            <Tooltip
              componentId="mlflow.mcp_registry.detail.remote.visibility_tooltip"
              content={
                isVisible ? (
                  <FormattedMessage
                    defaultMessage="Shown in Connect tab. Click to hide."
                    description="Tooltip for visible endpoint toggle"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Hidden from Connect tab. Click to show."
                    description="Tooltip for hidden endpoint toggle"
                  />
                )
              }
            >
              <Button
                componentId="mlflow.mcp_registry.detail.remote.visibility_row"
                type="tertiary"
                size="small"
                icon={isVisible ? <VisibleIcon /> : <VisibleOffIcon />}
                onClick={() => onToggleVisibility?.(!!isHidden)}
                aria-label={intl.formatMessage(
                  {
                    defaultMessage: '{action} endpoint {url}',
                    description: 'Aria label for remote endpoint visibility toggle',
                  },
                  { action: isVisible ? 'Hide' : 'Show', url: remote.url ?? remote.type },
                )}
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
              hasAdvancedDetails ? (
                <ViewDetailsDrawer title={remote.url ?? remote.type}>
                  <DetailField
                    label={intl.formatMessage({
                      defaultMessage: 'Transport type',
                      description: 'Remote transport type label',
                    })}
                    value={remote.type}
                    tagColor="default"
                  />
                  {remote.url && (
                    <DetailField
                      label={intl.formatMessage({ defaultMessage: 'URL', description: 'Remote URL label' })}
                      value={remote.url}
                      mono
                      link
                    />
                  )}
                  {remote.headers && remote.headers.length > 0 && (
                    <div>
                      <Typography.Text bold size="sm" css={blockLabelStyles(theme)}>
                        <FormattedMessage
                          defaultMessage="Headers ({count})"
                          description="MCP server remote headers heading with count"
                          values={{ count: remote.headers.length }}
                        />
                      </Typography.Text>
                      <div css={borderedListContainerStyles(theme)}>
                        {remote.headers.map((header, i) => (
                          <div key={header.name} css={borderedListItemStyles(theme, i > 0)}>
                            <div css={flexRowStyles(theme)}>
                              <Typography.Text bold size="sm" css={monoFontStyles}>
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
                      <Typography.Text bold size="sm" css={blockLabelStyles(theme)}>
                        <FormattedMessage
                          defaultMessage="URL Variables"
                          description="MCP server remote URL variables heading"
                        />
                      </Typography.Text>
                      <div css={borderedListContainerStyles(theme)}>
                        {Object.entries(remote.variables).map(([name, variable], i) => {
                          return (
                            <div key={name} css={borderedListItemStyles(theme, i > 0)}>
                              <Typography.Text bold size="sm" css={monoFontStyles}>
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
              ) : undefined
            }
          />
        </div>
      )}
    </div>
  );
};

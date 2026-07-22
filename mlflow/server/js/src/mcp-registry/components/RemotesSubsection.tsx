import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ConnectionSource } from '../types';
import type { ServerJSONPayload } from '../types';
import {
  borderedListContainerStyles,
  borderedListItemStyles,
  expandedContentPanelStyles,
  ellipsisStyles,
  flexRowStyles,
  noShrinkStyles,
  blockLabelStyles,
  monoFontStyles,
} from '../styles';
import { buildRemoteConnectOptionKey } from '../utils';
import { ViewDetailsDrawer, DetailField } from './ViewDetailsDrawer';
import { ConnectionInstructions } from './ConnectionInstructions';
import { SubsectionHelpHeading } from './SubsectionHelpHeading';
import { ExpandableListSection } from './ExpandableListSection';
import { VisibilityToggle } from './VisibilityToggle';

type Remote = NonNullable<ServerJSONPayload['remotes']>[number];

export const RemotesSubsection = ({
  remotes,
  derivedName,
  showVisibilityControls,
  connectOptions,
  onToggleConnectOption,
}: {
  remotes: Remote[];
  derivedName: string;
  showVisibilityControls?: boolean;
  connectOptions?: Record<string, { hidden?: boolean }>;
  onToggleConnectOption?: (key: string, visible: boolean) => void;
}) => {
  const intl = useIntl();
  const visibleRemotes = showVisibilityControls
    ? remotes
    : remotes.filter((r) => !connectOptions?.[buildRemoteConnectOptionKey(r)]?.hidden);

  if (visibleRemotes.length === 0) return null;

  return (
    <div>
      <SubsectionHelpHeading
        title={
          <FormattedMessage
            defaultMessage="Official endpoints"
            description="MCP server official endpoints subsection heading"
          />
        }
        componentId="mlflow.mcp_registry.detail.remotes_help"
        helpAriaLabel={intl.formatMessage({
          defaultMessage: 'About official endpoints',
          description: 'Aria label for remotes subsection help popover',
        })}
        helpText={
          <FormattedMessage
            defaultMessage="Remote endpoints provided by the server maintainer for direct connections."
            description="Help text for MCP server official endpoints subsection"
          />
        }
      />
      <ExpandableListSection
        items={visibleRemotes}
        getKey={(remote, index) => `${remote.type}-${remote.url ?? index}`}
        getAriaLabel={(remote, expanded) =>
          intl.formatMessage(
            {
              defaultMessage: '{action} remote {url}',
              description: 'Aria label for expanding/collapsing a remote row',
            },
            { action: expanded ? 'Collapse' : 'Expand', url: remote.url ?? remote.type },
          )
        }
        getRowStyle={(remote) =>
          connectOptions?.[buildRemoteConnectOptionKey(remote)]?.hidden ? { opacity: 0.5 } : undefined
        }
        renderRow={({ item: remote }) => (
          <RemoteRowContent
            remote={remote}
            showVisibilityControls={showVisibilityControls}
            isHidden={connectOptions?.[buildRemoteConnectOptionKey(remote)]?.hidden ?? false}
            onToggleVisibility={(visible) => onToggleConnectOption?.(buildRemoteConnectOptionKey(remote), visible)}
          />
        )}
        renderExpanded={(remote) => <RemoteExpandedContent remote={remote} derivedName={derivedName} />}
      />
    </div>
  );
};

const RemoteRowContent = ({
  remote,
  showVisibilityControls,
  isHidden,
  onToggleVisibility,
}: {
  remote: Remote;
  showVisibilityControls?: boolean;
  isHidden: boolean;
  onToggleVisibility?: (visible: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isVisible = !isHidden;

  return (
    <>
      <Tag componentId="mlflow.mcp_registry.detail.remote_transport_tag" color="charcoal" css={noShrinkStyles}>
        {remote.type}
      </Tag>
      <Typography.Text color="secondary" css={ellipsisStyles(theme)}>
        {remote.url ?? ''}
      </Typography.Text>
      {showVisibilityControls && (
        <VisibilityToggle
          componentId="mlflow.mcp_registry.detail.remote"
          isVisible={isVisible}
          onToggle={(nowVisible) => onToggleVisibility?.(nowVisible)}
          showDisabledTag
          ariaLabel={intl.formatMessage(
            {
              defaultMessage: '{action} endpoint {url}',
              description: 'Aria label for remote endpoint visibility toggle',
            },
            { action: isVisible ? 'Hide' : 'Show', url: remote.url ?? remote.type },
          )}
        />
      )}
    </>
  );
};

const RemoteExpandedContent = ({ remote, derivedName }: { remote: Remote; derivedName: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const hasAdvancedDetails =
    (remote.headers && remote.headers.length > 0) || (remote.variables && Object.keys(remote.variables).length > 0);

  return (
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
                  {Object.entries(remote.variables).map(([name, variable], i) => (
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
                  ))}
                </div>
              </div>
            )}
          </ViewDetailsDrawer>
        ) : undefined
      }
    />
  );
};

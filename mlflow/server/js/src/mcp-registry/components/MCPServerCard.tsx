import { useState } from 'react';
import { Button, Card, ConnectIcon, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPServer } from '../types';
import MCPRegistryRoutes from '../routes';
import { textClampStyles, textEllipsisStyles, cardBodyStyles, cardHeaderRowStyles, noShrinkStyles } from '../styles';
import { resolveDisplayName } from '../utils';
import { useServerState } from '../hooks/useServerState';
import { MCPServerIcon } from './MCPServerIcon';
import { MCPServerTags } from './MCPServerTags';
import { QuickConnectModal } from './QuickConnectModal';
import Utils from '../../common/utils/Utils';
import { useNavigate } from '../../common/utils/RoutingUtils';

export const MCPServerCard = ({ server }: { server: MCPServer }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();

  const [connectModalOpen, setConnectModalOpen] = useState(false);
  const timestamp = server.last_updated_timestamp
    ? Utils.formatTimestamp(server.last_updated_timestamp, intl)
    : undefined;
  const { isDimmed, isUnavailable } = useServerState(server);
  const hasTags = Object.keys(server.tags || {}).length > 0;

  return (
    <>
      <Card
        componentId="mlflow.mcp_registry.card"
        width="100%"
        navigateFn={async () => {
          navigate(MCPRegistryRoutes.getMCPServerDetailRoute(server.name));
        }}
        disableHover={isDimmed}
        dangerouslyAppendEmotionCSS={{
          height: '100%',
          '& > div': { display: 'flex', flexDirection: 'column', flexGrow: 1 },
          ...(isDimmed
            ? {
                cursor: 'pointer',
                '&:hover': {
                  borderColor: theme.colors.textSecondary,
                },
              }
            : {}),
        }}
      >
        <div css={{ ...cardBodyStyles(theme), opacity: isDimmed ? 0.5 : 1 }}>
          <div css={cardHeaderRowStyles(theme)}>
            <MCPServerIcon icons={server.icons} name={server.name} />
            <Typography.Text bold css={{ ...textEllipsisStyles, flex: 1 }}>
              {resolveDisplayName(server)}
            </Typography.Text>
            {server.latest_version && (
              <Typography.Text color="secondary" size="sm" css={noShrinkStyles}>
                v{server.latest_version}
              </Typography.Text>
            )}
          </div>
          {server.description && (
            <Typography.Text color="secondary" size="sm" css={textClampStyles(hasTags ? 2 : 3)}>
              {server.description}
            </Typography.Text>
          )}
          {hasTags && <MCPServerTags tags={server.tags || {}} />}
          <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 'auto' }}>
            {timestamp && (
              <Typography.Text color="secondary" size="sm">
                {timestamp}
              </Typography.Text>
            )}
            {!isUnavailable && (
              <Button
                componentId="mlflow.mcp_registry.card.connect"
                type="tertiary"
                size="small"
                icon={<ConnectIcon />}
                onClick={(e: React.MouseEvent) => {
                  e.stopPropagation();
                  e.preventDefault();
                  setConnectModalOpen(true);
                }}
                css={{ color: theme.colors.actionPrimaryBackgroundDefault }}
              />
            )}
            {isUnavailable && (
              <Tooltip
                componentId="mlflow.mcp_registry.card.bindings_tooltip"
                content={
                  <FormattedMessage
                    defaultMessage="No access endpoints configured"
                    description="Tooltip for disabled access endpoints icon on MCP server card"
                  />
                }
              >
                <span
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: theme.spacing.xs,
                    color: theme.colors.actionDisabledText,
                    cursor: 'not-allowed',
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    e.preventDefault();
                  }}
                >
                  <ConnectIcon css={{ width: 16, height: 16 }} />
                </span>
              </Tooltip>
            )}
          </div>
        </div>
      </Card>
      {connectModalOpen && (
        <QuickConnectModal visible={connectModalOpen} server={server} onClose={() => setConnectModalOpen(false)} />
      )}
    </>
  );
};

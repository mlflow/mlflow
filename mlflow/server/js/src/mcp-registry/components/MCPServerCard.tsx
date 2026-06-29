import { Card, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';

import type { MCPServer } from '../types';
import { resolveDisplayName } from '../utils';
import { textClampStyles, textEllipsisStyles, cardBodyStyles, cardHeaderRowStyles } from '../styles';
import { MCPServerIcon } from './MCPServerIcon';
import { MCPServerTags } from './MCPServerTags';
import Utils from '../../common/utils/Utils';

export const MCPServerCard = ({ server }: { server: MCPServer }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const displayName = resolveDisplayName(server);
  const timestamp = server.last_updated_timestamp
    ? Utils.formatTimestamp(server.last_updated_timestamp, intl)
    : undefined;

  return (
    <Card
      componentId="mlflow.mcp_registry.card"
      width="100%"
      role="button"
      onClick={() => {
        // TODO: navigate to server detail page once it exists
      }}
      dangerouslyAppendEmotionCSS={{ height: '100%', cursor: 'pointer' }}
    >
      <div css={cardBodyStyles(theme)}>
        <div css={cardHeaderRowStyles(theme)}>
          <MCPServerIcon icons={server.icons} name={displayName} />
          <Typography.Text bold css={{ ...textEllipsisStyles, flex: 1 }}>
            {displayName}
          </Typography.Text>
          {server.latest_version && (
            <Typography.Text color="secondary" size="sm" css={{ flexShrink: 0 }}>
              v{server.latest_version}
            </Typography.Text>
          )}
        </div>
        {server.description && (
          <Typography.Text color="secondary" size="sm" css={textClampStyles(2)}>
            {server.description}
          </Typography.Text>
        )}
        {Object.keys(server.tags || {}).length > 0 && <MCPServerTags tags={server.tags || {}} />}
        {timestamp && (
          <Typography.Text color="secondary" size="sm">
            {timestamp}
          </Typography.Text>
        )}
      </div>
    </Card>
  );
};

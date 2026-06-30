import type { CursorPaginationProps } from '@databricks/design-system';
import { CursorPagination, Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import type { MCPServer } from '../types';
import { MCPServerCard } from './MCPServerCard';
import { MCPServersEmptyState } from './MCPRegistryEmptyState';
import { cardGridStyles, flexColumnContainerStyles } from '../styles';

export const MCPServerCardGrid = ({
  servers,
  isLoading,
  isFiltered,
  hasNextPage,
  hasPreviousPage,
  onNextPage,
  onPreviousPage,
  pageSizeSelect,
}: {
  servers?: MCPServer[];
  isLoading?: boolean;
  isFiltered?: boolean;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  pageSizeSelect?: CursorPaginationProps['pageSizeSelect'];
}) => {
  const { theme } = useDesignSystemTheme();

  if (isLoading) {
    return (
      <div
        role="status"
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing.sm,
          padding: theme.spacing.lg,
          minHeight: 200,
        }}
      >
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading servers..." description="Loading state for MCP servers card grid" />
      </div>
    );
  }

  if (!servers?.length) {
    return <MCPServersEmptyState isFiltered={isFiltered} componentId="mlflow.mcp_registry.empty_state.create_server" />;
  }

  return (
    <div css={{ ...flexColumnContainerStyles, minHeight: 0 }}>
      <div role="list" aria-label="MCP servers" css={cardGridStyles(theme)}>
        {servers.map((server) => (
          <div role="listitem" key={server.name}>
            <MCPServerCard server={server} />
          </div>
        ))}
      </div>
      <div
        css={{
          flexShrink: 0,
          display: 'flex',
          justifyContent: 'flex-end',
          paddingTop: theme.spacing.sm,
          paddingBottom: theme.spacing.sm,
        }}
      >
        <CursorPagination
          hasNextPage={hasNextPage}
          hasPreviousPage={hasPreviousPage}
          onNextPage={onNextPage}
          onPreviousPage={onPreviousPage}
          pageSizeSelect={pageSizeSelect}
          componentId="mlflow.mcp_registry.grid.pagination"
        />
      </div>
    </div>
  );
};

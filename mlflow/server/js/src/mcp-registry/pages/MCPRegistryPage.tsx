import { useState } from 'react';
import {
  Alert,
  Button,
  GridIcon,
  Header,
  ListIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  McpIcon,
  Spacer,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useMCPServersListQuery } from '../hooks/useMCPServersListQuery';
import { useCreateMCPServerVersionModal } from '../hooks/useCreateMCPServerVersionModal';
import { MCPServerCardGrid } from '../components/MCPServerCardGrid';
import { MCPServerListTable } from '../components/MCPServerListTable';
import { MCPServerListFilters } from '../components/MCPServerListFilters';
import MCPRegistryRoutes from '../routes';
import { flexColumnContainerStyles, headerIconStyles } from '../styles';
import { MCPRegistryBetaTag } from '../components/MCPRegistryBetaTag';
import { useDebounce } from 'use-debounce';

type ViewMode = 'list' | 'grid';

const MCPRegistryPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [filterActive, setFilterActive] = useState(false);
  const [filterHasEndpoints, setFilterHasEndpoints] = useState(false);
  const [searchFilter, setSearchFilter] = useState('');
  const [debouncedSearchFilter] = useDebounce(searchFilter, 500);
  const navigate = useNavigate();

  const {
    data: servers,
    isLoading,
    error,
    hasNextPage,
    hasPreviousPage,
    onNextPage,
    onPreviousPage,
    pageSizeSelect,
    isFetching,
  } = useMCPServersListQuery({
    searchFilter: debouncedSearchFilter,
    filterActive,
    filterHasEndpoints,
  });

  const { CreateMCPServerVersionModal, openModal } = useCreateMCPServerVersionModal({
    onSuccess: ({ name }) => navigate(MCPRegistryRoutes.getMCPServerDetailRoute(name)),
  });

  const hasActiveFilters = Boolean(debouncedSearchFilter) || filterActive || filterHasEndpoints;
  const isServersEmpty = !isLoading && !isFetching && !error && !servers?.length && !hasActiveFilters;
  const createButton = !isServersEmpty ? (
    <Button componentId="mlflow.mcp_registry.create_server_button" type="primary" onClick={openModal}>
      <FormattedMessage defaultMessage="Create MCP server" description="Button to create a new MCP server" />
    </Button>
  ) : null;

  return (
    <>
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <Spacer shrinks={false} />
        <Header
          title={
            <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <span css={headerIconStyles(theme)}>
                <McpIcon />
              </span>
              <FormattedMessage defaultMessage="MCP Registry" description="MCP Registry page title" />
              <MCPRegistryBetaTag />
            </span>
          }
          buttons={createButton}
        />
        <Spacer shrinks={false} />
        <div css={flexColumnContainerStyles}>
          <div
            css={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: theme.spacing.sm,
              flexShrink: 0,
            }}
          >
            <div css={{ flex: 1 }}>
              <MCPServerListFilters
                searchFilter={searchFilter}
                onSearchFilterChange={setSearchFilter}
                filterActive={filterActive}
                onFilterActiveChange={setFilterActive}
                filterHasEndpoints={filterHasEndpoints}
                onFilterHasEndpointsChange={setFilterHasEndpoints}
                componentId="mlflow.mcp_registry.search"
              />
            </div>
            <SegmentedControlGroup
              name="mcp-registry-view-mode"
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as ViewMode)}
              componentId="mlflow.mcp_registry.view_toggle"
            >
              <SegmentedControlButton
                value="list"
                icon={<ListIcon />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'List view',
                  description: 'Aria label for list view toggle',
                })}
              />
              <SegmentedControlButton
                value="grid"
                icon={<GridIcon />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Grid view',
                  description: 'Aria label for grid view toggle',
                })}
              />
            </SegmentedControlGroup>
          </div>
          {error?.message && (
            <Alert
              type="error"
              message={error.message}
              componentId="mlflow.mcp_registry.error"
              closable={false}
              css={{ marginTop: theme.spacing.sm, flexShrink: 0 }}
            />
          )}
          {!error &&
            (viewMode === 'grid' ? (
              <MCPServerCardGrid
                servers={servers}
                isLoading={isLoading}
                isFiltered={hasActiveFilters}
                hasNextPage={hasNextPage}
                hasPreviousPage={hasPreviousPage}
                onNextPage={onNextPage}
                onPreviousPage={onPreviousPage}
                pageSizeSelect={pageSizeSelect}
                onCreateServer={openModal}
              />
            ) : (
              <MCPServerListTable
                servers={servers}
                hasNextPage={hasNextPage}
                hasPreviousPage={hasPreviousPage}
                isLoading={isLoading}
                isFiltered={hasActiveFilters}
                onNextPage={onNextPage}
                onPreviousPage={onPreviousPage}
                pageSizeSelect={pageSizeSelect}
                onCreateServer={openModal}
              />
            ))}
        </div>
      </ScrollablePageWrapper>
      {CreateMCPServerVersionModal}
    </>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.MCP_REGISTRY, MCPRegistryPage);

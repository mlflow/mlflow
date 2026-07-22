import { useState } from 'react';
import {
  Alert,
  Button,
  GridIcon,
  Header,
  ListIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  WrenchIcon,
  Spacer,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useNavigate } from '../../common/utils/RoutingUtils';
import { useIsAuthAvailable, useCurrentUserQuery } from '../../account/hooks';
import { useMCPServersListQuery } from '../hooks/useMCPServersListQuery';
import { useCreateMCPServerVersionModal } from '../hooks/useCreateMCPServerVersionModal';
import { MCPServerCardGrid } from '../components/MCPServerCardGrid';
import { MCPServerListTable } from '../components/MCPServerListTable';
import { MCPServerListFilters } from '../components/MCPServerListFilters';
import MCPRegistryRoutes from '../routes';
import { flexColumnContainerStyles, headerIconStyles } from '../styles';
import { useDebounce } from 'use-debounce';

type ViewMode = 'list' | 'grid';
type FilterMode = 'available' | 'all';

const MCPRegistryPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isAuthAvailable = useIsAuthAvailable();
  const { isLoading: isAuthLoading } = useCurrentUserQuery();

  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [filterMode, setFilterMode] = useState<FilterMode>(isAuthAvailable ? 'available' : 'all');
  const [searchFilter, setSearchFilter] = useState('');
  const [debouncedSearchFilter] = useDebounce(searchFilter, 500);
  const navigate = useNavigate();

  const effectiveFilterMode = !isAuthLoading && isAuthAvailable ? filterMode : 'all';

  const {
    data: servers,
    rawResponse,
    isLoading,
    error,
    hasNextPage,
    hasPreviousPage,
    onNextPage,
    onPreviousPage,
    pageSizeSelect,
  } = useMCPServersListQuery({
    searchFilter: debouncedSearchFilter,
    availableOnly: effectiveFilterMode === 'available',
  });

  const { CreateMCPServerVersionModal, openModal } = useCreateMCPServerVersionModal({
    onSuccess: ({ name }) => navigate(MCPRegistryRoutes.getMCPServerDetailRoute(name)),
  });

  const hasManageOnAny = rawResponse !== undefined && rawResponse.user_has_manage !== false;
  const showAvailabilityFilter = !isAuthLoading && isAuthAvailable && hasManageOnAny;
  const isServersEmpty = !isLoading && !error && !servers?.length && !debouncedSearchFilter;
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
                <WrenchIcon />
              </span>
              <FormattedMessage defaultMessage="MCP Registry" description="MCP Registry page title" />
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
                componentId="mlflow.mcp_registry.search"
              />
            </div>
            {showAvailabilityFilter && (
              <SegmentedControlGroup
                name="mcp-registry-filter-mode"
                value={filterMode}
                onChange={(e) => setFilterMode(e.target.value as FilterMode)}
                componentId="mlflow.mcp_registry.filter_toggle"
                data-testid="mcp-registry-availability-filter"
              >
                <SegmentedControlButton value="available">
                  <FormattedMessage defaultMessage="Available" description="Filter to show only available servers" />
                </SegmentedControlButton>
                <SegmentedControlButton value="all">
                  <FormattedMessage defaultMessage="All" description="Filter to show all servers" />
                </SegmentedControlButton>
              </SegmentedControlGroup>
            )}
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
                isFiltered={Boolean(debouncedSearchFilter)}
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
                isFiltered={Boolean(debouncedSearchFilter)}
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

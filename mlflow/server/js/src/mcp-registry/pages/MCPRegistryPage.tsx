import { useCallback, useState } from 'react';
import {
  Alert,
  Button,
  GridIcon,
  Header,
  ListIcon,
  PlusIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  WrenchIcon,
  Spacer,
  Table,
  TableHeader,
  TableRow,
  TableFilterInput,
  TableFilterLayout,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { RadioChangeEvent } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useSearchParams } from '../../common/utils/RoutingUtils';
import { useMCPServersListQuery } from '../hooks/useMCPServersListQuery';
import { MCPServerCardGrid } from '../components/MCPServerCardGrid';
import { MCPServerListTable } from '../components/MCPServerListTable';
import { MCPServerListFilters } from '../components/MCPServerListFilters';
import { MCPRegistryEmptyState } from '../components/MCPRegistryEmptyState';
import { flexColumnContainerStyles, headerIconStyles } from '../styles';
import { useDebounce } from 'use-debounce';

type ViewMode = 'list' | 'grid';
type ActiveTab = 'servers' | 'bindings';

const MCPRegistryPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [searchParams, setSearchParams] = useSearchParams();
  const tabFromUrl = searchParams.get('tab');
  const activeTab: ActiveTab = tabFromUrl === 'bindings' ? 'bindings' : 'servers';
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [searchFilter, setSearchFilter] = useState('');
  const [debouncedSearchFilter] = useDebounce(searchFilter, 500);

  const {
    data: servers,
    isLoading,
    error,
    hasNextPage,
    hasPreviousPage,
    onNextPage,
    onPreviousPage,
    pageSizeSelect,
  } = useMCPServersListQuery({
    searchFilter: activeTab === 'servers' ? debouncedSearchFilter : undefined,
    enabled: activeTab === 'servers',
  });

  const handleTabChange = useCallback(
    (e: RadioChangeEvent) => {
      const value = e.target.value as ActiveTab;
      setSearchFilter('');
      const next = new URLSearchParams(searchParams);
      if (value === 'servers') {
        next.delete('tab');
      } else {
        next.set('tab', value);
      }
      setSearchParams(next, { replace: true });
    },
    [searchParams, setSearchParams],
  );

  const hideCreateButton = !isLoading && !servers?.length && !debouncedSearchFilter;
  const createButton = !hideCreateButton ? (
    <Button componentId="mlflow.mcp_registry.create_server_button" type="primary" disabled>
      <FormattedMessage defaultMessage="Create MCP server" description="Button to create a new MCP server" />
    </Button>
  ) : null;

  return (
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
        <SegmentedControlGroup
          name="mcp-registry-tabs"
          value={activeTab}
          onChange={handleTabChange}
          componentId="mlflow.mcp_registry.tabs"
        >
          <SegmentedControlButton value="servers">
            <FormattedMessage defaultMessage="Servers" description="MCP Registry servers tab label" />
          </SegmentedControlButton>
          <SegmentedControlButton value="bindings">
            <FormattedMessage defaultMessage="Access Bindings" description="MCP Registry access bindings tab label" />
          </SegmentedControlButton>
        </SegmentedControlGroup>

        {activeTab === 'servers' && (
          <div css={flexColumnContainerStyles}>
            <div
              css={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: theme.spacing.sm,
                paddingTop: theme.spacing.md,
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
                />
              ))}
          </div>
        )}

        {activeTab === 'bindings' && (
          <>
            <div css={{ paddingTop: theme.spacing.md }}>
              <TableFilterLayout>
                <TableFilterInput
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Search access bindings',
                    description: 'Placeholder for MCP access bindings search filter input',
                  })}
                  componentId="mlflow.mcp_registry.bindings.search"
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                />
              </TableFilterLayout>
            </div>
            <Table
              scrollable
              empty={
                <MCPRegistryEmptyState
                  title={
                    <FormattedMessage
                      defaultMessage="Create access binding"
                      description="Empty state title for MCP access bindings tab"
                    />
                  }
                  description={
                    <FormattedMessage
                      defaultMessage="Create and manage access bindings for your MCP servers."
                      description="Empty state description for MCP access bindings tab"
                    />
                  }
                  button={
                    <Button
                      componentId="mlflow.mcp_registry.bindings.empty_state.create"
                      type="primary"
                      icon={<PlusIcon />}
                      disabled
                    >
                      <FormattedMessage
                        defaultMessage="Create access binding"
                        description="MCP Registry bindings empty state CTA button"
                      />
                    </Button>
                  }
                />
              }
            >
              <TableRow isHeader>
                <TableHeader componentId="mlflow.mcp_registry.bindings.header.endpoint">
                  <FormattedMessage defaultMessage="Endpoint" description="Access bindings table header for endpoint" />
                </TableHeader>
                <TableHeader componentId="mlflow.mcp_registry.bindings.header.server">
                  <FormattedMessage
                    defaultMessage="MCP Server"
                    description="Access bindings table header for server name"
                  />
                </TableHeader>
                <TableHeader componentId="mlflow.mcp_registry.bindings.header.version">
                  <FormattedMessage
                    defaultMessage="Version/Alias"
                    description="Access bindings table header for version or alias"
                  />
                </TableHeader>
                <TableHeader componentId="mlflow.mcp_registry.bindings.header.transport">
                  <FormattedMessage
                    defaultMessage="Transport"
                    description="Access bindings table header for transport type"
                  />
                </TableHeader>
                <TableHeader componentId="mlflow.mcp_registry.bindings.header.last_updated">
                  <FormattedMessage
                    defaultMessage="Last updated"
                    description="Access bindings table header for last updated"
                  />
                </TableHeader>
              </TableRow>
            </Table>
          </>
        )}
      </div>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.MCP_REGISTRY, MCPRegistryPage);

import { useCallback, useState } from 'react';
import {
  Button,
  Empty,
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

type ViewMode = 'list' | 'grid';
type ActiveTab = 'servers' | 'bindings';

const emptyCenterStyles = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100%',
  minHeight: 400,
  width: '100%',
  '& > div': {
    height: '100%',
    display: 'flex',
    flexDirection: 'column' as const,
    justifyContent: 'center',
    alignItems: 'center',
  },
};

const MCPRegistryPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [searchParams, setSearchParams] = useSearchParams();
  const tabFromUrl = searchParams.get('tab');
  const activeTab: ActiveTab = tabFromUrl === 'bindings' ? 'bindings' : 'servers';
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [searchFilter, setSearchFilter] = useState('');

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

  // TODO: show create button only when server list is non-empty (matches PromptsPage pattern)
  const isEmptyState = true;
  const createButton = !isEmptyState ? (
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
            <span
              css={{
                display: 'flex',
                borderRadius: theme.borders.borderRadiusSm,
                backgroundColor: theme.colors.backgroundSecondary,
                padding: theme.spacing.sm,
              }}
            >
              <WrenchIcon />
            </span>
            <FormattedMessage defaultMessage="MCP Registry" description="MCP Registry page title" />
          </span>
        }
        buttons={createButton}
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
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
          <>
            <div
              css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm, paddingTop: theme.spacing.md }}
            >
              <div css={{ flex: 1 }}>
                <TableFilterLayout>
                  <TableFilterInput
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Search MCP servers by name',
                      description: 'Placeholder for MCP server search filter input',
                    })}
                    componentId="mlflow.mcp_registry.search"
                    value={searchFilter}
                    onChange={(e) => setSearchFilter(e.target.value)}
                  />
                </TableFilterLayout>
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
            <Table
              scrollable
              empty={
                <div css={emptyCenterStyles}>
                  <Empty
                    title={
                      <FormattedMessage
                        defaultMessage="Create MCP server"
                        description="Empty state title for MCP servers tab"
                      />
                    }
                    description={
                      <FormattedMessage
                        defaultMessage="Create and manage MCP servers using MLflow."
                        description="Empty state description for MCP servers tab"
                      />
                    }
                    button={
                      <Button
                        componentId="mlflow.mcp_registry.empty_state.create_server"
                        type="primary"
                        icon={<PlusIcon />}
                        disabled
                      >
                        <FormattedMessage
                          defaultMessage="Create MCP server"
                          description="MCP Registry empty state CTA button"
                        />
                      </Button>
                    }
                  />
                </div>
              }
            >
              {viewMode === 'list' && (
                <TableRow isHeader>
                  <TableHeader componentId="mlflow.mcp_registry.table.header.name">
                    <FormattedMessage defaultMessage="Name" description="MCP servers table header for name column" />
                  </TableHeader>
                  <TableHeader componentId="mlflow.mcp_registry.table.header.description">
                    <FormattedMessage
                      defaultMessage="Description"
                      description="MCP servers table header for description column"
                    />
                  </TableHeader>
                  <TableHeader componentId="mlflow.mcp_registry.table.header.last_modified">
                    <FormattedMessage
                      defaultMessage="Last modified"
                      description="MCP servers table header for last modified column"
                    />
                  </TableHeader>
                </TableRow>
              )}
            </Table>
          </>
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
                <div css={emptyCenterStyles}>
                  <Empty
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
                </div>
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

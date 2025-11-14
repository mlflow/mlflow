import {
  Button,
  ChevronDownIcon,
  Empty,
  FilterIcon,
  Header,
  Input,
  PlusIcon,
  Popover,
  SearchIcon,
  Spacer,
  Spinner,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
// eslint-disable-next-line import/no-extraneous-dependencies
import { notification } from 'antd';
import { useState, useCallback, useMemo, useEffect } from 'react';
import type { SortingState } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import LocalStorageUtils from '@mlflow/mlflow/src/common/utils/LocalStorageUtils';
import { ExperimentListViewTagsFilter } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/components/ExperimentListViewTagsFilter';
import { useListRoutes } from '../hooks/useListRoutes';
import { useListSecrets } from '../hooks/useListSecrets';
import { useCreateRoute } from '../hooks/useCreateRoute';
import { useUpdateRoute } from '../hooks/useUpdateRoute';
import { useRoutesTagsFilter } from '../hooks/useRoutesTagsFilter';
import { CreateRouteModal } from './CreateRouteModal';
import { AddRouteModal } from './AddRouteModal';
import { RoutesTable } from './RoutesTable';
import { RouteDetailDrawer } from './RouteDetailDrawer';
import { UpdateRouteModal } from './UpdateRouteModal';
import type { Route } from '../types';

const DEFAULT_HIDDEN_COLUMNS = ['tags', 'created_at', 'created_by', 'last_updated_by'];
const LOCAL_STORAGE_KEY = 'hiddenColumns';

export default function RoutesPage() {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { routes = [], isLoading, error } = useListRoutes({ enabled: true });
  const { secrets = [], refetch: refetchSecrets } = useListSecrets({ enabled: true });
  const { createRouteAsync } = useCreateRoute();
  const { updateRouteAsync } = useUpdateRoute();
  const { tagsFilter, setTagsFilter, isTagsFilterOpen, setIsTagsFilterOpen } = useRoutesTagsFilter();
  const [showCreateRouteModal, setShowCreateRouteModal] = useState(false);
  const [showAddRouteModal, setShowAddRouteModal] = useState(false);
  const [showUpdateRouteModal, setShowUpdateRouteModal] = useState(false);
  const [selectedRoute, setSelectedRoute] = useState<Route | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [sorting, setSorting] = useState<SortingState>([
    { id: 'name', desc: false },
    { id: 'last_updated_at', desc: true },
  ]);

  // Initialize localStorage store for routes
  const localStorageStore = useMemo(() => LocalStorageUtils.getStoreForComponent('RoutesPage', 'default'), []);

  // Load hidden columns from localStorage or use defaults
  const [hiddenColumns, setHiddenColumns] = useState<string[]>(() => {
    const savedColumns = localStorageStore.getItem(LOCAL_STORAGE_KEY);
    if (savedColumns) {
      try {
        return JSON.parse(savedColumns);
      } catch (e) {
        return DEFAULT_HIDDEN_COLUMNS;
      }
    }
    return DEFAULT_HIDDEN_COLUMNS;
  });

  const [searchText, setSearchText] = useState('');

  // Save hidden columns to localStorage whenever they change
  useEffect(() => {
    localStorageStore.setItem(LOCAL_STORAGE_KEY, JSON.stringify(hiddenColumns));
  }, [hiddenColumns, localStorageStore]);

  const toggleHiddenColumn = useCallback((columnId: string) => {
    setHiddenColumns((prev) => {
      if (prev.includes(columnId)) {
        return prev.filter((id) => id !== columnId);
      }
      return [...prev, columnId];
    });
  }, []);

  const filteredRoutes = useMemo(() => {
    return routes.filter((route) => {
      const matchesSearch = searchText
        ? (route.name && route.name.toLowerCase().includes(searchText.toLowerCase())) ||
          route.route_id.toLowerCase().includes(searchText.toLowerCase()) ||
          route.model_name.toLowerCase().includes(searchText.toLowerCase()) ||
          route.provider.toLowerCase().includes(searchText.toLowerCase())
        : true;

      const matchesTags =
        tagsFilter.length === 0 ||
        tagsFilter.every((filter) => {
          if (!route.tags) return false;

          // Convert tags to array format
          const tagEntries = Array.isArray(route.tags)
            ? route.tags
            : Object.entries(route.tags).map(([key, value]) => ({ key, value }));

          // Find matching tag by key
          const matchingTag = tagEntries.find((tag) => tag.key === filter.key);
          if (!matchingTag) return false;

          // Apply operator
          const tagValue = matchingTag.value.toLowerCase();
          const filterValue = filter.value.toLowerCase();

          switch (filter.operator) {
            case 'IS':
              return tagValue === filterValue;
            case 'IS NOT':
              return tagValue !== filterValue;
            case 'CONTAINS':
              return tagValue.includes(filterValue);
            default:
              return false;
          }
        });

      return matchesSearch && matchesTags;
    });
  }, [routes, searchText, tagsFilter]);

  const parseErrorMessage = (error: any): { title: string; description: string } => {
    const errorMsg = error.message || error.error_message || String(error);

    // Handle binding conflicts - this error should be handled in the modal to highlight the field
    if (errorMsg.includes('Binding already exists')) {
      return {
        title: intl.formatMessage({
          defaultMessage: 'Route name already exists',
          description: 'Routes page > binding conflict error title',
        }),
        description: intl.formatMessage({
          defaultMessage: 'Please select a different route name.',
          description: 'Routes page > binding conflict error description',
        }),
      };
    }

    // Handle secret name conflicts
    if (errorMsg.includes('Secret') && errorMsg.includes('already exists')) {
      return {
        title: intl.formatMessage({
          defaultMessage: 'Key name already exists',
          description: 'Routes page > secret conflict error title',
        }),
        description: intl.formatMessage({
          defaultMessage: 'A key with this name already exists. Please choose a different key name.',
          description: 'Routes page > secret conflict error description',
        }),
      };
    }

    // Handle authentication/authorization errors
    if (errorMsg.includes('401') || errorMsg.includes('Unauthorized') || errorMsg.includes('authentication')) {
      return {
        title: intl.formatMessage({
          defaultMessage: 'Authentication failed',
          description: 'Routes page > auth error title',
        }),
        description: intl.formatMessage({
          defaultMessage: 'Invalid API key. Please check your credentials and try again.',
          description: 'Routes page > auth error description',
        }),
      };
    }

    // Default error
    return {
      title: intl.formatMessage({
        defaultMessage: 'Failed to create route',
        description: 'Routes page > generic error title',
      }),
      description:
        errorMsg ||
        intl.formatMessage({
          defaultMessage: 'An unexpected error occurred. Please try again.',
          description: 'Routes page > generic error description',
        }),
    };
  };

  const handleCreateRoute = async (routeData: any) => {
    try {
      await createRouteAsync(routeData);
      // Refetch secrets since a new secret was created
      refetchSecrets();
      notification.success({
        message: intl.formatMessage({
          defaultMessage: 'Route created successfully',
          description: 'Routes page > create route success notification',
        }),
      });
      setShowCreateRouteModal(false);
    } catch (err: any) {
      const errorMsg = err.message || err.error_message || String(err);

      // Don't show notification for binding conflicts - the modal handles these
      if (!errorMsg.includes('Binding already exists')) {
        const { title, description } = parseErrorMessage(err);
        notification.error({
          message: title,
          description,
        });
      }
      throw err;
    }
  };

  const handleAddRoute = async (routeData: any) => {
    try {
      await createRouteAsync(routeData);
      notification.success({
        message: intl.formatMessage({
          defaultMessage: 'Route added successfully',
          description: 'Routes page > add route success notification',
        }),
      });
      setShowAddRouteModal(false);
    } catch (err: any) {
      const errorMsg = err.message || err.error_message || String(err);

      // Don't show notification for binding conflicts - the modal handles these
      if (!errorMsg.includes('Binding already exists')) {
        const { title, description } = parseErrorMessage(err);
        notification.error({
          message: title,
          description,
        });
      }
      throw err;
    }
  };

  const handleRowClick = (route: Route) => {
    setSelectedRoute(route);
    setIsDrawerOpen(true);
  };

  const handleDrawerClose = () => {
    setIsDrawerOpen(false);
    // Small delay before clearing selected route to avoid drawer content flickering
    setTimeout(() => setSelectedRoute(null), 200);
  };

  const handleUpdateRoute = (route: Route) => {
    setSelectedRoute(route);
    setIsDrawerOpen(false);
    setShowUpdateRouteModal(true);
  };

  const handleUpdateRouteSubmit = async (
    routeId: string,
    updateData: {
      secret_id?: string;
      secret_name?: string;
      secret_value?: string;
      provider?: string;
      auth_config?: string;
    },
  ) => {
    if (!selectedRoute) return;

    try {
      await updateRouteAsync({
        route_id: routeId,
        ...updateData,
      });

      // Refetch secrets if we created a new one
      if (updateData.secret_name) {
        refetchSecrets();
      }

      notification.success({
        message: intl.formatMessage({
          defaultMessage: 'Route updated successfully',
          description: 'Routes page > update route success notification',
        }),
      });
      setShowUpdateRouteModal(false);
    } catch (err: any) {
      const errorMsg = err.message || err.error_message || String(err);
      notification.error({
        message: intl.formatMessage({
          defaultMessage: 'Failed to update route',
          description: 'Routes page > update route error title',
        }),
        description: errorMsg,
      });
      throw err;
    }
  };

  const handleDeleteRoute = (route: Route) => {
    // TODO: Implement delete route confirmation modal
    setIsDrawerOpen(false);
    notification.info({
      message: intl.formatMessage({
        defaultMessage: 'Delete route functionality coming soon',
        description: 'Routes page > delete route placeholder notification',
      }),
    });
  };

  if (isLoading) {
    return (
      <ScrollablePageWrapper>
        <Spacer shrinks={false} />
        <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
          <Spinner />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (error) {
    return (
      <ScrollablePageWrapper>
        <Spacer shrinks={false} />
        <div css={{ padding: theme.spacing.lg }}>
          <Typography.Text color="error">
            <FormattedMessage defaultMessage="Failed to load routes" description="Routes page > error loading routes" />
          </Typography.Text>
        </div>
      </ScrollablePageWrapper>
    );
  }

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={<FormattedMessage defaultMessage="Routes" description="Header title for the routes page" />}
        breadcrumbs={[]}
        buttons={
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Tooltip
              componentId="mlflow.routes.add_route_button_tooltip"
              content={
                secrets.length === 0
                  ? intl.formatMessage({
                      defaultMessage: 'A route needs to be created first',
                      description: 'Routes page > add route button disabled tooltip',
                    })
                  : undefined
              }
            >
              <span>
                <Button
                  componentId="mlflow.routes.add_route_button"
                  onClick={() => setShowAddRouteModal(true)}
                  icon={<PlusIcon />}
                  disabled={secrets.length === 0}
                >
                  <FormattedMessage
                    defaultMessage="Add Route"
                    description="Add route button label (use existing secret)"
                  />
                </Button>
              </span>
            </Tooltip>
            <Button
              componentId="mlflow.routes.create_route_button"
              type="primary"
              onClick={() => setShowCreateRouteModal(true)}
              icon={<PlusIcon />}
            >
              <FormattedMessage defaultMessage="Create Route" description="Create route button label (new secret)" />
            </Button>
          </div>
        }
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: theme.spacing.lg }}>
        {routes.length === 0 ? (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: theme.spacing.md,
              minHeight: 400,
            }}
          >
            <Empty
              title={
                <FormattedMessage
                  defaultMessage="No routes yet"
                  description="Routes page > no routes empty state title"
                />
              }
              description={
                <FormattedMessage
                  defaultMessage="Create a route to configure model access with API keys. Routes connect secrets to models and can be bound to specific resources."
                  description="Routes page > no routes empty state description"
                />
              }
            />
            <Button
              componentId="mlflow.routes.empty_state.create_route_button"
              type="primary"
              onClick={() => setShowCreateRouteModal(true)}
              icon={<PlusIcon />}
            >
              <FormattedMessage
                defaultMessage="Create Route"
                description="Routes page > empty state create route button"
              />
            </Button>
          </div>
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, flex: 1, overflow: 'hidden' }}>
            <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
              <Input
                componentId="mlflow.routes.search_input"
                prefix={<SearchIcon />}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Search routes by name',
                  description: 'Routes page > search input placeholder',
                })}
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                allowClear
                css={{ width: 300 }}
              />
              <Popover.Root
                componentId="mlflow.routes.tag_filter"
                open={isTagsFilterOpen}
                onOpenChange={setIsTagsFilterOpen}
              >
                <Popover.Trigger asChild>
                  <Button
                    componentId="mlflow.routes.tag_filter.trigger"
                    icon={<FilterIcon />}
                    endIcon={<ChevronDownIcon />}
                    type={tagsFilter.length > 0 ? 'primary' : undefined}
                  >
                    <FormattedMessage
                      defaultMessage="Tag filter"
                      description="Button to open the tags filter popover in the routes page"
                    />
                  </Button>
                </Popover.Trigger>
                <Popover.Content>
                  <ExperimentListViewTagsFilter tagsFilter={tagsFilter} setTagsFilter={setTagsFilter} />
                </Popover.Content>
              </Popover.Root>
            </div>
            <RoutesTable
              routes={filteredRoutes}
              loading={isLoading}
              error={error || undefined}
              sorting={sorting}
              setSorting={setSorting}
              hiddenColumns={hiddenColumns}
              toggleHiddenColumn={toggleHiddenColumn}
              onRowClick={handleRowClick}
            />
          </div>
        )}
      </div>

      <CreateRouteModal
        visible={showCreateRouteModal}
        onCancel={() => setShowCreateRouteModal(false)}
        onCreate={handleCreateRoute}
      />
      <AddRouteModal
        visible={showAddRouteModal}
        onCancel={() => setShowAddRouteModal(false)}
        onCreate={handleAddRoute}
        onOpenCreateModal={() => {
          setShowAddRouteModal(false);
          setShowCreateRouteModal(true);
        }}
        availableSecrets={secrets}
      />

      <RouteDetailDrawer
        route={selectedRoute}
        open={isDrawerOpen}
        onClose={handleDrawerClose}
        onUpdate={handleUpdateRoute}
        onDelete={handleDeleteRoute}
      />

      <UpdateRouteModal
        route={selectedRoute}
        visible={showUpdateRouteModal}
        onCancel={() => setShowUpdateRouteModal(false)}
        onUpdate={handleUpdateRouteSubmit}
      />
    </ScrollablePageWrapper>
  );
}

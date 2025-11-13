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
import { useCreateRoute } from '../hooks/useCreateRoute';
import { useRoutesTagsFilter } from '../hooks/useRoutesTagsFilter';
import { CreateRouteModal } from './CreateRouteModal';
import { RoutesTable } from './RoutesTable';

const DEFAULT_HIDDEN_COLUMNS = ['tags', 'created_at', 'created_by', 'last_updated_by'];
const LOCAL_STORAGE_KEY = 'hiddenColumns';

export default function RoutesPage() {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { routes = [], isLoading, error } = useListRoutes({ enabled: true });
  const { createRouteAsync } = useCreateRoute();
  const { tagsFilter, setTagsFilter, isTagsFilterOpen, setIsTagsFilterOpen } = useRoutesTagsFilter();
  const [showCreateRouteModal, setShowCreateRouteModal] = useState(false);
  const [showAddRouteModal, setShowAddRouteModal] = useState(false);
  const [sorting, setSorting] = useState<SortingState>([
    { id: 'secret_name', desc: false },
    { id: 'created_at', desc: true }
  ]);

  // Initialize localStorage store for routes
  const localStorageStore = useMemo(
    () => LocalStorageUtils.getStoreForComponent('RoutesPage', 'default'),
    []
  );

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

      const matchesTags = tagsFilter.length === 0 || tagsFilter.every((filter) => {
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
      description: errorMsg || intl.formatMessage({
        defaultMessage: 'An unexpected error occurred. Please try again.',
        description: 'Routes page > generic error description',
      }),
    };
  };

  const handleCreateRoute = async (routeData: any) => {
    try {
      await createRouteAsync(routeData);
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
            <FormattedMessage
              defaultMessage="Failed to load routes"
              description="Routes page > error loading routes"
            />
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
            <Button
              componentId="mlflow.routes.add_route_button"
              onClick={() => setShowAddRouteModal(true)}
              icon={<PlusIcon />}
            >
              <FormattedMessage defaultMessage="Add Route" description="Add route button label (use existing secret)" />
            </Button>
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
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No routes yet. Create a route to get started."
                description="Routes page > no routes empty state"
              />
            }
          />
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
            />
          </div>
        )}
      </div>

      <CreateRouteModal
        visible={showCreateRouteModal}
        onCancel={() => setShowCreateRouteModal(false)}
        onCreate={handleCreateRoute}
      />
      {/* TODO: Add AddRouteModal */}
    </ScrollablePageWrapper>
  );
}

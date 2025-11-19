import {
  Button,
  ChevronDownIcon,
  Empty,
  FilterIcon,
  GearIcon,
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
import { useDebounce } from 'use-debounce';
import type { SortingState } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import LocalStorageUtils from '@mlflow/mlflow/src/common/utils/LocalStorageUtils';
import { ExperimentListViewTagsFilter } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/components/ExperimentListViewTagsFilter';
import { useBackendSupport } from '@mlflow/mlflow/src/common/hooks/useBackendSupport';
import { useListEndpoints } from '../hooks/useListEndpoints';
import { useListSecrets } from '../hooks/useListSecrets';
import { useCreateEndpoint } from '../hooks/useCreateEndpoint';
import { useUpdateEndpoint } from '../hooks/useUpdateEndpoint';
import { useDeleteEndpointMutation } from '../hooks/useDeleteEndpointMutation';
import { useRoutesTagsFilter } from '../hooks/useRoutesTagsFilter';
import { CreateRouteModal } from './CreateRouteModal';
import { AddRouteModal } from './AddRouteModal';
import { RoutesTable } from './RoutesTable';
import { RouteDetailDrawer } from './RouteDetailDrawer';
import { UpdateRouteModal } from './UpdateRouteModal';
import { SecretManagementDrawer } from './SecretManagementDrawer';
import { GatewayRequiresSqlBackend } from './GatewayRequiresSqlBackend';
import type { Endpoint } from '../types';

const DEFAULT_HIDDEN_COLUMNS = ['tags', 'created_at', 'created_by', 'last_updated_by'];
const LOCAL_STORAGE_KEY = 'hiddenColumns';

export default function RoutesPage() {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { isSqlBackend, storeType, isLoading: isCheckingBackend } = useBackendSupport();
  const { endpoints = [], isLoading, error } = useListEndpoints({ enabled: isSqlBackend === true });
  const { secrets = [], refetch: refetchSecrets } = useListSecrets({ enabled: true });
  const { createEndpointAsync } = useCreateEndpoint();
  const { updateEndpointAsync } = useUpdateEndpoint();
  const { tagsFilter, setTagsFilter, isTagsFilterOpen, setIsTagsFilterOpen } = useRoutesTagsFilter();
  const { deleteEndpoint, isLoading: isDeleting } = useDeleteEndpointMutation({
    onSuccess: () => {
      setIsDrawerOpen(false);
      notification.success({
        message: intl.formatMessage({
          defaultMessage: 'Endpoint deleted successfully',
          description: 'Endpoints page > delete endpoint success',
        }),
      });
    },
    onError: (error) => {
      notification.error({
        message: intl.formatMessage({
          defaultMessage: 'Failed to delete endpoint',
          description: 'Endpoints page > delete endpoint error',
        }),
        description: error.message,
      });
    },
  });
  const [showCreateRouteModal, setShowCreateRouteModal] = useState(false);
  const [showAddRouteModal, setShowAddRouteModal] = useState(false);
  const [showUpdateRouteModal, setShowUpdateRouteModal] = useState(false);
  const [showManagementDrawer, setShowManagementDrawer] = useState(false);
  const [selectedEndpoint, setSelectedEndpoint] = useState<Endpoint | null>(null);
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
  const [debouncedSearchText] = useDebounce(searchText, 500);

  // Save hidden columns to localStorage whenever they change
  useEffect(() => {
    localStorageStore.setItem(LOCAL_STORAGE_KEY, JSON.stringify(hiddenColumns));
  }, [hiddenColumns, localStorageStore]);

  // Update selectedEndpoint when endpoints data changes (e.g., after updating an endpoint)
  useEffect(() => {
    if (selectedEndpoint && endpoints.length > 0) {
      const updatedEndpoint = endpoints.find((r) => r.endpoint_id === selectedEndpoint.endpoint_id);
      if (updatedEndpoint) {
        setSelectedEndpoint(updatedEndpoint);
      }
    }
  }, [endpoints, selectedEndpoint]);

  const toggleHiddenColumn = useCallback((columnId: string) => {
    setHiddenColumns((prev) => {
      if (prev.includes(columnId)) {
        return prev.filter((id) => id !== columnId);
      }
      return [...prev, columnId];
    });
  }, []);

  const filteredEndpoints = useMemo(() => {
    return endpoints.filter((endpoint) => {
      const matchesSearch = debouncedSearchText
        ? (endpoint.name && endpoint.name.toLowerCase().includes(debouncedSearchText.toLowerCase())) ||
          endpoint.endpoint_id.toLowerCase().includes(debouncedSearchText.toLowerCase()) ||
          endpoint.model_name.toLowerCase().includes(debouncedSearchText.toLowerCase()) ||
          (endpoint.provider && endpoint.provider.toLowerCase().includes(debouncedSearchText.toLowerCase()))
        : true;

      const matchesTags =
        tagsFilter.length === 0 ||
        tagsFilter.every((filter) => {
          if (!endpoint.tags) return false;

          // Convert tags to array format
          const tagEntries = Array.isArray(endpoint.tags)
            ? endpoint.tags
            : Object.entries(endpoint.tags).map(([key, value]) => ({ key, value }));

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
  }, [endpoints, debouncedSearchText, tagsFilter]);

  // Show placeholder if FileStore is detected
  if (isCheckingBackend) {
    return (
      <ScrollablePageWrapper>
        <div css={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
          <Spinner />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (isSqlBackend === false) {
    return <GatewayRequiresSqlBackend storeType={storeType} />;
  }

  const parseErrorMessage = (error: any): { title: string; description: string } => {
    const errorMsg = error.message || error.error_message || String(error);

    // Handle binding conflicts - this error should be handled in the modal to highlight the field
    if (errorMsg.includes('Binding already exists')) {
      return {
        title: intl.formatMessage({
          defaultMessage: 'Endpoint name already exists',
          description: 'Endpoints page > binding conflict error title',
        }),
        description: intl.formatMessage({
          defaultMessage: 'Please select a different endpoint name.',
          description: 'Endpoints page > binding conflict error description',
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
        defaultMessage: 'Failed to create endpoint',
        description: 'Endpoints page > generic error title',
      }),
      description:
        errorMsg ||
        intl.formatMessage({
          defaultMessage: 'An unexpected error occurred. Please try again.',
          description: 'Endpoints page > generic error description',
        }),
    };
  };

  const handleCreateRoute = async (routeData: any) => {
    try {
      await createEndpointAsync(routeData);
      // Refetch secrets since a new secret was created
      refetchSecrets();
      notification.success({
        message: intl.formatMessage({
          defaultMessage: 'Endpoint created successfully',
          description: 'Endpoints page > create endpoint success notification',
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
      await createEndpointAsync(routeData);
      notification.success({
        message: intl.formatMessage({
          defaultMessage: 'Endpoint added successfully',
          description: 'Endpoints page > add endpoint success notification',
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

  const handleRowClick = (endpoint: Endpoint) => {
    setSelectedEndpoint(endpoint);
    setIsDrawerOpen(true);
  };

  const handleDrawerClose = () => {
    setIsDrawerOpen(false);
    // Small delay before clearing selected endpoint to avoid drawer content flickering
    setTimeout(() => setSelectedEndpoint(null), 200);
  };

  const handleUpdateRoute = (endpoint: Endpoint) => {
    setSelectedEndpoint(endpoint);
    setIsDrawerOpen(false);
    setShowUpdateRouteModal(true);
  };

  const handleUpdateRouteSubmit = async (
    endpointId: string,
    updateData: {
      secret_id?: string;
      secret_name?: string;
      secret_value?: string;
      provider?: string;
      auth_config?: string;
      route_description?: string;
      route_tags?: string;
    },
  ) => {
    if (!selectedEndpoint) return;

    try {
      await updateEndpointAsync({
        endpoint_id: endpointId,
        ...updateData,
      });

      // Refetch secrets if we created a new one
      if (updateData.secret_name) {
        refetchSecrets();
      }

      notification.success({
        message: intl.formatMessage({
          defaultMessage: 'Endpoint updated successfully',
          description: 'Endpoints page > update endpoint success notification',
        }),
      });
      setShowUpdateRouteModal(false);
    } catch (err: any) {
      const errorMsg = err.message || err.error_message || String(err);
      notification.error({
        message: intl.formatMessage({
          defaultMessage: 'Failed to update endpoint',
          description: 'Endpoints page > update endpoint error title',
        }),
        description: errorMsg,
      });
      throw err;
    }
  };

  const handleDeleteRoute = (endpoint: Endpoint) => {
    deleteEndpoint(endpoint.endpoint_id);
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
            <FormattedMessage defaultMessage="Failed to load endpoints" description="Endpoints page > error loading endpoints" />
          </Typography.Text>
        </div>
      </ScrollablePageWrapper>
    );
  }

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={<FormattedMessage defaultMessage="Endpoints" description="Header title for the endpoints page" />}
        breadcrumbs={[]}
        buttons={
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Button
              componentId="mlflow.routes.manage_secrets_button"
              icon={<GearIcon />}
              onClick={() => setShowManagementDrawer(true)}
            >
              <FormattedMessage defaultMessage="Manage Secrets" description="Manage secrets button label" />
            </Button>
            <Tooltip
              componentId="mlflow.routes.add_route_button_tooltip"
              content={
                secrets.length === 0
                  ? intl.formatMessage({
                      defaultMessage: 'An endpoint needs to be created first',
                      description: 'Endpoints page > add endpoint button disabled tooltip',
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
                    defaultMessage="Add Endpoint"
                    description="Add endpoint button label (use existing secret)"
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
              <FormattedMessage defaultMessage="Create Endpoint" description="Create endpoint button label (new secret)" />
            </Button>
          </div>
        }
      />
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: theme.spacing.lg }}>
        {endpoints.length === 0 ? (
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
                  defaultMessage="No endpoints yet"
                  description="Endpoints page > no endpoints empty state title"
                />
              }
              description={
                <FormattedMessage
                  defaultMessage="Create an endpoint to configure model access with API keys. Endpoints connect secrets to models and can be bound to specific resources."
                  description="Endpoints page > no endpoints empty state description"
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
                defaultMessage="Create Endpoint"
                description="Endpoints page > empty state create endpoint button"
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
                  defaultMessage: 'Search endpoints by name',
                  description: 'Endpoints page > search input placeholder',
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
                      description="Button to open the tags filter popover in the endpoints page"
                    />
                  </Button>
                </Popover.Trigger>
                <Popover.Content>
                  <ExperimentListViewTagsFilter tagsFilter={tagsFilter} setTagsFilter={setTagsFilter} />
                </Popover.Content>
              </Popover.Root>
            </div>
            <RoutesTable
              routes={filteredEndpoints}
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
        route={selectedEndpoint}
        open={isDrawerOpen}
        onClose={handleDrawerClose}
        onUpdate={handleUpdateRouteSubmit}
        onDelete={handleDeleteRoute}
      />

      <UpdateRouteModal
        route={selectedEndpoint}
        visible={showUpdateRouteModal}
        onCancel={() => setShowUpdateRouteModal(false)}
        onUpdate={handleUpdateRouteSubmit}
      />

      <SecretManagementDrawer open={showManagementDrawer} onClose={() => setShowManagementDrawer(false)} />
    </ScrollablePageWrapper>
  );
}

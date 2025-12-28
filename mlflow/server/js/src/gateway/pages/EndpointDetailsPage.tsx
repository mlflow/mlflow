import {
  Alert,
  Breadcrumb,
  Button,
  Card,
  PencilIcon,
  Spinner,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useParams, Link } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import GatewayRoutes from '../routes';
import { TimeAgo } from '../../shared/web-shared/browse/TimeAgo';
import { useEndpointDetails } from '../hooks/useEndpointDetails';
import { ModelCard } from '../components/endpoint-details/ModelCard';
import { ConnectedResourcesSection } from '../components/endpoint-details/ConnectedResourcesSection';
import { DeleteEndpointModal } from '../components/endpoints/DeleteEndpointModal';
import { ApiKeyDetailsDrawer } from '../components/api-keys/ApiKeyDetailsDrawer';
import type { EndpointModelMapping, ProviderModel, SecretInfo } from '../types';
import { useState, useCallback } from 'react';

const EndpointDetailsPage = () => {
  const { theme } = useDesignSystemTheme();
  const { endpointId } = useParams<{ endpointId: string }>();

  const {
    endpoint,
    modelsData,
    endpointBindings,
    hasModels,
    isLoading,
    error,
    isDeleteModalOpen,
    handleEdit,
    handleDeleteClick,
    handleDeleteModalClose,
    handleDeleteSuccess,
  } = useEndpointDetails(endpointId ?? '');

  const [selectedSecret, setSelectedSecret] = useState<SecretInfo | null>(null);
  const handleKeyClick = useCallback((secret: SecretInfo) => {
    setSelectedSecret(secret);
  }, []);
  const handleKeyDrawerClose = useCallback(() => {
    setSelectedSecret(null);
  }, []);

  if (isLoading) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading endpoint..." description="Loading message for endpoint" />
        </div>
      </div>
    );
  }

  if (error || !endpoint) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="mlflow.gateway.endpoint-details.error"
            type="error"
            message={(error as Error | null)?.message ?? 'Endpoint not found'}
          />
        </div>
      </div>
    );
  }

  return (
    <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
      <div css={{ padding: theme.spacing.md }}>
        <Breadcrumb includeTrailingCaret>
          <Breadcrumb.Item>
            <Link to={GatewayRoutes.gatewayPageRoute}>
              <FormattedMessage defaultMessage="AI Gateway" description="Breadcrumb link to gateway page" />
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Item>
            <Link to={GatewayRoutes.gatewayPageRoute}>
              <FormattedMessage defaultMessage="Endpoints" description="Breadcrumb link to endpoints list" />
            </Link>
          </Breadcrumb.Item>
        </Breadcrumb>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginTop: theme.spacing.sm,
          }}
        >
          <Typography.Title level={2}>{endpoint.name ?? endpoint.endpoint_id}</Typography.Title>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Button componentId="mlflow.gateway.endpoint-details.edit" icon={<PencilIcon />} onClick={handleEdit}>
              <FormattedMessage
                defaultMessage="Edit"
                description="Gateway > Endpoint details page > Edit endpoint button"
              />
            </Button>
            <Button
              componentId="mlflow.gateway.endpoint-details.delete"
              danger
              icon={<TrashIcon />}
              onClick={handleDeleteClick}
            >
              <FormattedMessage
                defaultMessage="Delete"
                description="Gateway > Endpoint details page > Delete endpoint button"
              />
            </Button>
          </div>
        </div>
        <div
          css={{
            marginTop: theme.spacing.md,
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        />
      </div>

      <div
        css={{
          flex: 1,
          display: 'flex',
          gap: theme.spacing.md,
          padding: `0 ${theme.spacing.md}px ${theme.spacing.md}px`,
          overflow: 'auto',
        }}
      >
        <div css={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          <Card componentId="mlflow.gateway.endpoint-details.config-card" css={{ width: '100%' }}>
            <div css={{ padding: theme.spacing.md, width: '100%' }}>
              <Typography.Title level={3}>
                <FormattedMessage defaultMessage="Active configuration" description="Section title for active config" />
              </Typography.Title>

              {hasModels ? (
                <div
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.md,
                    marginTop: theme.spacing.lg,
                    width: '100%',
                  }}
                >
                  <div css={{ width: '100%' }}>
                    <Typography.Text bold color="secondary" css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
                      <FormattedMessage defaultMessage="Models" description="Models section label" />
                    </Typography.Text>
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, width: '100%' }}>
                      {endpoint.model_mappings.map((mapping: EndpointModelMapping) => (
                        <ModelCard
                          key={mapping.mapping_id}
                          modelDefinition={mapping.model_definition}
                          modelMetadata={modelsData?.find(
                            (m: ProviderModel) => m.model === mapping.model_definition?.model_name,
                          )}
                          onKeyClick={handleKeyClick}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <Typography.Text color="secondary">
                  <FormattedMessage
                    defaultMessage="No models configured for this endpoint"
                    description="Message when no models configured"
                  />
                </Typography.Text>
              )}
            </div>
          </Card>
        </div>

        <div css={{ width: 300, flexShrink: 0 }}>
          <Card componentId="mlflow.gateway.endpoint-details.about-card">
            <div css={{ padding: theme.spacing.md }}>
              <Typography.Title level={4} css={{ marginBottom: theme.spacing.md }}>
                <FormattedMessage defaultMessage="About this endpoint" description="Sidebar title" />
              </Typography.Title>

              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                <div>
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Created" description="Created at label" />
                  </Typography.Text>
                  <div css={{ marginTop: theme.spacing.xs }}>
                    <TimeAgo date={new Date(endpoint.created_at)} />
                  </div>
                </div>

                <div>
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Last modified" description="Last modified label" />
                  </Typography.Text>
                  <div css={{ marginTop: theme.spacing.xs }}>
                    <TimeAgo date={new Date(endpoint.last_updated_at)} />
                  </div>
                </div>

                {endpoint.created_by && (
                  <div>
                    <Typography.Text color="secondary">
                      <FormattedMessage defaultMessage="Created by" description="Created by label" />
                    </Typography.Text>
                    <div css={{ marginTop: theme.spacing.xs }}>
                      <Typography.Text>{endpoint.created_by}</Typography.Text>
                    </div>
                  </div>
                )}

                <ConnectedResourcesSection bindings={endpointBindings} />
              </div>
            </div>
          </Card>
        </div>
      </div>

      <DeleteEndpointModal
        open={isDeleteModalOpen}
        endpoint={endpoint}
        bindings={endpointBindings}
        onClose={handleDeleteModalClose}
        onSuccess={handleDeleteSuccess}
      />

      <ApiKeyDetailsDrawer open={selectedSecret !== null} secret={selectedSecret} onClose={handleKeyDrawerClose} />
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, EndpointDetailsPage);

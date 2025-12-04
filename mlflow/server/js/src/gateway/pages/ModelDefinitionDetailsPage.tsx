import { useState } from 'react';
import { useParams, Link, useNavigate } from '../../common/utils/RoutingUtils';
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
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import GatewayRoutes from '../routes';
import { formatProviderName } from '../utils/providerUtils';
import { timestampToDate } from '../utils/dateUtils';
import { TimeAgo } from '../../shared/web-shared/browse/TimeAgo';
import { useModelDefinitionQuery } from '../hooks/useModelDefinitionQuery';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import { useSecretQuery } from '../hooks/useSecretQuery';
import { DeleteModelDefinitionModal } from '../components/model-definitions/DeleteModelDefinitionModal';
import type { Endpoint } from '../types';

const ModelDefinitionDetailsPage = () => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const { modelDefinitionId } = useParams<{ modelDefinitionId: string }>();

  const { data, error, isLoading, refetch } = useModelDefinitionQuery(modelDefinitionId ?? '');
  const modelDefinition = data?.model_definition;

  const { data: endpoints } = useEndpointsQuery();
  const { data: secretData } = useSecretQuery(modelDefinition?.secret_id);
  const secret = secretData?.secret;

  const [showDeleteModal, setShowDeleteModal] = useState(false);

  // Get endpoints using this model definition
  const boundEndpoints = (endpoints ?? []).filter((endpoint: Endpoint) =>
    endpoint.model_mappings?.some((mapping) => mapping.model_definition_id === modelDefinitionId),
  );

  const handleEdit = () => {
    navigate(GatewayRoutes.getEditModelDefinitionRoute(modelDefinitionId ?? ''));
  };

  const handleDeleteClick = () => {
    setShowDeleteModal(true);
  };

  const handleDeleteSuccess = () => {
    navigate(GatewayRoutes.modelDefinitionsPageRoute);
  };

  if (isLoading) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading model..." description="Loading message for model" />
        </div>
      </div>
    );
  }

  if (error || !modelDefinition) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="mlflow.gateway.model-definition-details.error"
            type="error"
            message={(error as Error | null)?.message ?? 'Model not found'}
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
              <FormattedMessage defaultMessage="Gateway" description="Breadcrumb link to gateway page" />
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Item>
            <Link to={GatewayRoutes.modelDefinitionsPageRoute}>
              <FormattedMessage defaultMessage="Models" description="Breadcrumb link to models page" />
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
          <Typography.Title level={2}>{modelDefinition.name}</Typography.Title>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Button
              componentId="mlflow.gateway.model-definition-details.edit"
              type="tertiary"
              icon={<PencilIcon />}
              onClick={handleEdit}
            >
              <FormattedMessage defaultMessage="Edit" description="Edit model button" />
            </Button>
            <Button
              componentId="mlflow.gateway.model-definition-details.delete"
              type="tertiary"
              danger
              icon={<TrashIcon />}
              onClick={handleDeleteClick}
            >
              <FormattedMessage defaultMessage="Delete" description="Delete model button" />
            </Button>
          </div>
        </div>
      </div>

      <div
        css={{
          flex: 1,
          display: 'flex',
          gap: theme.spacing.lg,
          padding: `0 ${theme.spacing.md}px ${theme.spacing.md}px`,
          overflow: 'auto',
        }}
      >
        {/* Main content */}
        <div css={{ flex: 2, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {/* Model Configuration */}
          <Card componentId="mlflow.gateway.model-definition-details.config-card">
            <div css={{ padding: theme.spacing.md }}>
              <Typography.Title level={3}>
                <FormattedMessage defaultMessage="Model configuration" description="Section title for model config" />
              </Typography.Title>

              <div
                css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, marginTop: theme.spacing.lg }}
              >
                {/* Provider */}
                <div>
                  <Typography.Text bold color="secondary" css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
                    <FormattedMessage defaultMessage="Provider" description="Provider label" />
                  </Typography.Text>
                  <Typography.Text>{formatProviderName(modelDefinition.provider)}</Typography.Text>
                </div>

                {/* Model Name */}
                <div>
                  <Typography.Text bold color="secondary" css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
                    <FormattedMessage defaultMessage="Model" description="Model name label" />
                  </Typography.Text>
                  <Typography.Text css={{ fontFamily: 'monospace' }}>{modelDefinition.model_name}</Typography.Text>
                </div>

                {/* API Key */}
                <div>
                  <Typography.Text bold color="secondary" css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
                    <FormattedMessage defaultMessage="API Key" description="API key label" />
                  </Typography.Text>
                  {secret ? (
                    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                      <Typography.Text>{secret.secret_name}</Typography.Text>
                      <Typography.Text
                        css={{
                          fontFamily: 'monospace',
                          fontSize: theme.typography.fontSizeSm,
                          backgroundColor: theme.colors.tagDefault,
                          padding: `2px ${theme.spacing.xs}px`,
                          borderRadius: theme.general.borderRadiusBase,
                        }}
                      >
                        {secret.masked_value}
                      </Typography.Text>
                    </div>
                  ) : (
                    <Typography.Text>{modelDefinition.secret_name}</Typography.Text>
                  )}
                </div>
              </div>
            </div>
          </Card>

          {/* Endpoints Using This Model */}
          <Card componentId="mlflow.gateway.model-definition-details.endpoints-card">
            <div css={{ padding: theme.spacing.md }}>
              <Typography.Title level={3}>
                <FormattedMessage
                  defaultMessage="Endpoints using this model ({count})"
                  description="Section title for endpoints with count"
                  values={{ count: boundEndpoints.length }}
                />
              </Typography.Title>

              <div css={{ marginTop: theme.spacing.lg }}>
                {boundEndpoints.length === 0 ? (
                  <Typography.Text color="secondary">
                    <FormattedMessage
                      defaultMessage="This model is not attached to any endpoints yet."
                      description="Message when model is not used by any endpoints"
                    />
                  </Typography.Text>
                ) : (
                  <div
                    css={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: theme.spacing.sm,
                      maxHeight: boundEndpoints.length > 5 ? 200 : undefined,
                      overflowY: boundEndpoints.length > 5 ? 'auto' : undefined,
                    }}
                  >
                    {boundEndpoints.map((endpoint: Endpoint) => (
                      <Link
                        key={endpoint.endpoint_id}
                        to={GatewayRoutes.getEndpointDetailsRoute(endpoint.endpoint_id)}
                        css={{
                          color: theme.colors.actionPrimaryBackgroundDefault,
                          textDecoration: 'none',
                          '&:hover': { textDecoration: 'underline' },
                        }}
                      >
                        {endpoint.name ?? endpoint.endpoint_id}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </Card>
        </div>

        {/* Sidebar */}
        <div css={{ flex: 1, maxWidth: 300 }}>
          <Card componentId="mlflow.gateway.model-definition-details.about-card">
            <div css={{ padding: theme.spacing.md }}>
              <Typography.Title level={4} css={{ marginBottom: theme.spacing.md }}>
                <FormattedMessage defaultMessage="About this model" description="Sidebar title" />
              </Typography.Title>

              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                <div>
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Created" description="Created at label" />
                  </Typography.Text>
                  <div css={{ marginTop: theme.spacing.xs }}>
                    <TimeAgo date={timestampToDate(modelDefinition.created_at)} />
                  </div>
                </div>

                <div>
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Last modified" description="Last modified label" />
                  </Typography.Text>
                  <div css={{ marginTop: theme.spacing.xs }}>
                    <TimeAgo date={timestampToDate(modelDefinition.last_updated_at)} />
                  </div>
                </div>

                {modelDefinition.created_by && (
                  <div>
                    <Typography.Text color="secondary">
                      <FormattedMessage defaultMessage="Created by" description="Created by label" />
                    </Typography.Text>
                    <div css={{ marginTop: theme.spacing.xs }}>
                      <Typography.Text>{modelDefinition.created_by}</Typography.Text>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      <DeleteModelDefinitionModal
        open={showDeleteModal}
        modelDefinition={modelDefinition}
        endpoints={boundEndpoints}
        onClose={() => setShowDeleteModal(false)}
        onSuccess={handleDeleteSuccess}
      />
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, ModelDefinitionDetailsPage);

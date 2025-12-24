import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { FormProvider } from 'react-hook-form';
import { Breadcrumb, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { useCreateEndpointForm } from '../hooks/useCreateEndpointForm';
import { getReadableErrorMessage } from '../utils/errorUtils';
import { EndpointFormRenderer } from '../components/endpoint-form';
import GatewayRoutes from '../routes';

const CreateEndpointPage = () => {
  const { theme } = useDesignSystemTheme();
  const {
    form,
    isLoading,
    error,
    resetErrors,
    selectedModel,
    isFormComplete,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  } = useCreateEndpointForm();

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <FormProvider {...form}>
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
          <Typography.Title level={2} css={{ marginTop: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Create endpoint" description="Page title for create endpoint" />
          </Typography.Title>
          <div
            css={{
              marginTop: theme.spacing.md,
              borderBottom: `1px solid ${theme.colors.border}`,
            }}
          />
        </div>
        <EndpointFormRenderer
          mode="create"
          isSubmitting={isLoading}
          error={error}
          errorMessage={getReadableErrorMessage(error)}
          resetErrors={resetErrors}
          selectedModel={selectedModel}
          isFormComplete={isFormComplete}
          onSubmit={handleSubmit}
          onCancel={handleCancel}
          onNameBlur={handleNameBlur}
        />
      </FormProvider>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, CreateEndpointPage);

import { Link } from '../../../common/utils/RoutingUtils';
import {
  Alert,
  Breadcrumb,
  Button,
  Spinner,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Controller, UseFormReturn } from 'react-hook-form';
import { useState } from 'react';
import GatewayRoutes from '../../routes';
import { LongFormSummary } from '../../../common/components/long-form/LongFormSummary';
import type { EditEndpointFormData } from '../../hooks/useEditEndpointForm';
import { TrafficSplitConfigurator } from './TrafficSplitConfigurator';
import { FallbackModelsConfigurator } from './FallbackModelsConfigurator';
import { EndpointUsageModal } from '../endpoints/EndpointUsageModal';
import { EditableEndpointName } from './EditableEndpointName';
import type { Endpoint } from '../../types';

export interface EditEndpointFormRendererProps {
  form: UseFormReturn<EditEndpointFormData>;
  isLoadingEndpoint: boolean;
  isSubmitting: boolean;
  loadError: Error | null;
  mutationError: Error | null;
  errorMessage: string | null;
  endpoint: Endpoint | undefined;
  existingEndpoints: Endpoint[] | undefined;
  isFormComplete: boolean;
  hasChanges: boolean;
  onSubmit: (values: EditEndpointFormData) => Promise<void>;
  onCancel: () => void;
  onNameUpdate: (newName: string) => Promise<void>;
}

export const EditEndpointFormRenderer = ({
  form,
  isLoadingEndpoint,
  isSubmitting,
  loadError,
  mutationError,
  errorMessage,
  endpoint,
  existingEndpoints,
  isFormComplete,
  hasChanges,
  onSubmit,
  onCancel,
  onNameUpdate,
}: EditEndpointFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isUsageModalOpen, setIsUsageModalOpen] = useState(false);

  const trafficSplitModels = form.watch('trafficSplitModels');
  const fallbackModels = form.watch('fallbackModels');

  const totalWeight = trafficSplitModels.reduce((sum, m) => sum + m.weight, 0);
  const isValidTotal = Math.abs(totalWeight - 100) < 0.01;

  if (isLoadingEndpoint) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading endpoint..." description="Loading message for endpoint" />
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="mlflow.gateway.edit-endpoint.error"
            type="error"
            message={loadError.message ?? 'Endpoint not found'}
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
          css={{ marginTop: theme.spacing.sm, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
        >
          <EditableEndpointName
            endpoint={endpoint}
            existingEndpoints={existingEndpoints}
            onNameUpdate={onNameUpdate}
            isSubmitting={isSubmitting}
          />
          <Button componentId="mlflow.gateway.edit-endpoint.use-button" onClick={() => setIsUsageModalOpen(true)}>
            <FormattedMessage defaultMessage="Use" description="Use endpoint button" />
          </Button>
        </div>
      </div>

      {mutationError && (
        <div css={{ padding: `0 ${theme.spacing.md}px` }}>
          <Alert
            componentId="mlflow.gateway.edit-endpoint.mutation-error"
            closable={false}
            message={errorMessage}
            type="error"
            css={{ marginBottom: theme.spacing.md }}
          />
        </div>
      )}

      <div
        css={{
          flex: 1,
          display: 'flex',
          gap: theme.spacing.md,
          padding: `${theme.spacing.md}px`,
          overflow: 'auto',
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      >
        <div css={{ flex: 1, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          <div
            css={{
              padding: theme.spacing.md,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusMd,
              backgroundColor: theme.colors.backgroundSecondary,
            }}
          >
            <Typography.Title level={3}>
              <FormattedMessage
                defaultMessage="Priority 1 (Traffic Split)"
                description="Section title for traffic split"
              />
            </Typography.Title>
            <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
              <FormattedMessage
                defaultMessage="Models in this priority will be tested first, with traffic split load balancing"
                description="Traffic split description"
              />
            </Typography.Text>

            <div css={{ marginTop: theme.spacing.lg }}>
              <Controller
                control={form.control}
                name="trafficSplitModels"
                render={({ field }) => (
                  <TrafficSplitConfigurator
                    value={field.value}
                    onChange={field.onChange}
                    componentIdPrefix="mlflow.gateway.edit-endpoint.traffic-split"
                  />
                )}
              />
            </div>
          </div>

          <div
            css={{
              padding: theme.spacing.md,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusMd,
              backgroundColor: theme.colors.backgroundSecondary,
            }}
          >
            <Typography.Title level={3}>
              <FormattedMessage
                defaultMessage="Priority 2 (Fallback)"
                description="Section title for fallback models"
              />
            </Typography.Title>
            <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
              <FormattedMessage
                defaultMessage="Models in this priority will be tested second, after models in Priority 1 have failed. Models will be attempted in order from top to bottom."
                description="Fallback models description"
              />
            </Typography.Text>

            <div css={{ marginTop: theme.spacing.lg }}>
              <Controller
                control={form.control}
                name="fallbackModels"
                render={({ field }) => (
                  <FallbackModelsConfigurator
                    value={field.value}
                    onChange={field.onChange}
                    componentIdPrefix="mlflow.gateway.edit-endpoint.fallback"
                  />
                )}
              />
            </div>
          </div>

          <div css={{ display: 'flex', gap: theme.spacing.md }}>
            <div
              css={{
                flex: 1,
                padding: theme.spacing.md,
                border: `2px dashed ${theme.colors.actionDefaultBorderDefault}`,
                borderRadius: theme.borders.borderRadiusMd,
                backgroundColor: theme.colors.backgroundPrimary,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                textAlign: 'center',
              }}
            >
              <Typography.Text bold>
                <FormattedMessage defaultMessage="Usage Tracking" description="Section title for usage tracking" />
              </Typography.Text>
              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                <FormattedMessage defaultMessage="Coming Soon" description="Coming soon label" />
              </Typography.Text>
            </div>

            <div
              css={{
                flex: 1,
                padding: theme.spacing.md,
                border: `2px dashed ${theme.colors.actionDefaultBorderDefault}`,
                borderRadius: theme.borders.borderRadiusMd,
                backgroundColor: theme.colors.backgroundPrimary,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                textAlign: 'center',
              }}
            >
              <Typography.Text bold>
                <FormattedMessage defaultMessage="Rate Limiting" description="Section title for rate limiting" />
              </Typography.Text>
              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                <FormattedMessage defaultMessage="Coming Soon" description="Coming soon label" />
              </Typography.Text>
            </div>
          </div>
        </div>

        <div
          css={{
            width: 280,
            flexShrink: 0,
            position: 'sticky',
            top: 0,
            alignSelf: 'flex-start',
          }}
        >
          <LongFormSummary
            title={intl.formatMessage({
              defaultMessage: 'Summary',
              description: 'Summary sidebar title',
            })}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              {trafficSplitModels.length > 0 && (
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold color="secondary">
                    <FormattedMessage defaultMessage="Traffic Split" description="Summary traffic split label" />
                  </Typography.Text>
                  {trafficSplitModels.map((model, idx) => (
                    <div
                      key={idx}
                      css={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: theme.spacing.xs,
                        fontSize: theme.typography.fontSizeSm,
                      }}
                    >
                      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
                        {model.provider && model.modelName
                          ? `${model.provider}/${model.modelName}`
                          : `Model ${idx + 1}`}{' '}
                        - {model.weight}%
                      </Typography.Text>
                    </div>
                  ))}
                </div>
              )}

              {fallbackModels.length > 0 && (
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold color="secondary">
                    <FormattedMessage defaultMessage="Fallback Models" description="Summary fallback models label" />
                  </Typography.Text>
                  {fallbackModels.map((model, idx) => (
                    <div
                      key={idx}
                      css={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: theme.spacing.xs,
                        fontSize: theme.typography.fontSizeSm,
                      }}
                    >
                      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
                        {idx + 1}.{' '}
                        {model.provider && model.modelName
                          ? `${model.provider}/${model.modelName}`
                          : `Model ${idx + 1}`}
                      </Typography.Text>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </LongFormSummary>
        </div>
      </div>

      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          gap: theme.spacing.sm,
          padding: theme.spacing.md,
          borderTop: `1px solid ${theme.colors.border}`,
          flexShrink: 0,
        }}
      >
        <Button componentId="mlflow.gateway.edit-endpoint.cancel" onClick={onCancel}>
          <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
        </Button>
        <Tooltip
          componentId="mlflow.gateway.edit-endpoint.save-tooltip"
          content={
            !isFormComplete && trafficSplitModels.length > 0 && !isValidTotal
              ? intl.formatMessage({
                  defaultMessage: 'Traffic split percentages must total 100%',
                  description: 'Tooltip shown when save button is disabled due to invalid traffic split total',
                })
              : !isFormComplete
                ? intl.formatMessage({
                    defaultMessage: 'Please configure at least one model in traffic split',
                    description: 'Tooltip shown when save button is disabled due to incomplete form',
                  })
                : !hasChanges
                  ? intl.formatMessage({
                      defaultMessage: 'No changes to save',
                      description: 'Tooltip shown when save button is disabled due to no changes',
                    })
                  : undefined
          }
        >
          <Button
            componentId="mlflow.gateway.edit-endpoint.save"
            type="primary"
            onClick={form.handleSubmit(onSubmit)}
            loading={isSubmitting}
            disabled={!isFormComplete || !hasChanges}
          >
            <FormattedMessage defaultMessage="Save changes" description="Save changes button" />
          </Button>
        </Tooltip>
      </div>

      <EndpointUsageModal
        open={isUsageModalOpen}
        onClose={() => setIsUsageModalOpen(false)}
        endpointName={endpoint?.name || ''}
      />
    </div>
  );
};

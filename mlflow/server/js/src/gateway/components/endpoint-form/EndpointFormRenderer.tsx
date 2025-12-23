import { Alert, Button, FormUI, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { GatewayInput } from '../common';
import { FormattedMessage, useIntl } from 'react-intl';
import { Controller, useFormContext } from 'react-hook-form';
import { ProviderSelect } from '../create-endpoint/ProviderSelect';
import { formatProviderName } from '../../utils/providerUtils';

export interface EndpointFormData {
  name: string;
  provider: string;
}

export interface EndpointFormRendererProps {
  mode: 'create' | 'edit';
  isSubmitting: boolean;
  error: Error | null;
  errorMessage: string | null;
  resetErrors: () => void;
  isFormComplete: boolean;
  hasChanges?: boolean;
  onSubmit: (values: EndpointFormData) => Promise<void>;
  onCancel: () => void;
  onNameBlur: () => void;
  componentIdPrefix?: string;
}

export const EndpointFormRenderer = ({
  mode,
  isSubmitting,
  error,
  errorMessage,
  resetErrors,
  isFormComplete,
  hasChanges = true,
  onSubmit,
  onCancel,
  onNameBlur,
  componentIdPrefix = `mlflow.gateway.${mode}-endpoint`,
}: EndpointFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const form = useFormContext<EndpointFormData>();

  const provider = form.watch('provider');

  const isButtonDisabled = mode === 'edit' ? !isFormComplete || !hasChanges : !isFormComplete;
  const buttonTooltip = !isFormComplete
    ? intl.formatMessage({
        defaultMessage: 'Please complete all required fields',
        description: 'Tooltip shown when submit button is disabled due to incomplete form',
      })
    : mode === 'edit' && !hasChanges
    ? intl.formatMessage({
        defaultMessage: 'No changes to save',
        description: 'Tooltip shown when save button is disabled due to no changes',
      })
    : undefined;

  return (
    <>
      {error && (
        <div css={{ padding: `0 ${theme.spacing.md}px` }}>
          <Alert
            componentId={`${componentIdPrefix}.error`}
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
          gap: theme.spacing.lg,
          padding: theme.spacing.md,
          overflow: 'auto',
        }}
      >
        <div css={{ flexGrow: 1, maxWidth: 600 }}>
          <div css={{ marginBottom: theme.spacing.lg }}>
            <FormUI.Label htmlFor={`${componentIdPrefix}.name`}>
              <FormattedMessage defaultMessage="Name" description="Endpoint name label" />
            </FormUI.Label>
            <Controller
              control={form.control}
              name="name"
              rules={{ required: 'Name is required' }}
              render={({ field, fieldState }) => (
                <div>
                  <GatewayInput
                    id={`${componentIdPrefix}.name`}
                    componentId={`${componentIdPrefix}.name`}
                    {...field}
                    onChange={(e) => {
                      field.onChange(e);
                      form.clearErrors('name');
                      resetErrors();
                    }}
                    onBlur={() => {
                      field.onBlur();
                      onNameBlur();
                    }}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'my-endpoint',
                      description: 'Placeholder for endpoint name input',
                    })}
                    validationState={fieldState.error ? 'error' : undefined}
                  />
                  {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
                </div>
              )}
            />
          </div>

          <div css={{ marginBottom: theme.spacing.lg }}>
            <Controller
              control={form.control}
              name="provider"
              rules={{ required: 'Provider is required' }}
              render={({ field, fieldState }) => (
                <ProviderSelect
                  value={field.value}
                  onChange={field.onChange}
                  error={fieldState.error?.message}
                  componentIdPrefix={`${componentIdPrefix}.provider`}
                />
              )}
            />
          </div>
        </div>

        <div
          css={{
            flexShrink: 0,
            width: 300,
            padding: theme.spacing.md,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.general.borderRadiusBase,
          }}
        >
          <Typography.Title level={4} css={{ marginBottom: theme.spacing.md }}>
            <FormattedMessage defaultMessage="Summary" description="Summary sidebar title" />
          </Typography.Title>

          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <div>
              <Typography.Text bold color="secondary">
                <FormattedMessage defaultMessage="Provider" description="Summary provider label" />
              </Typography.Text>
              <Typography.Text css={{ display: 'block' }}>
                {provider ? (
                  formatProviderName(provider)
                ) : (
                  <span css={{ color: theme.colors.textSecondary }}>
                    <FormattedMessage defaultMessage="Not selected" description="Summary not selected" />
                  </span>
                )}
              </Typography.Text>
            </div>
          </div>
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
        <Button componentId={`${componentIdPrefix}.cancel`} onClick={onCancel}>
          <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
        </Button>
        <Tooltip componentId={`${componentIdPrefix}.submit-tooltip`} content={buttonTooltip}>
          <Button
            componentId={`${componentIdPrefix}.submit`}
            type="primary"
            onClick={form.handleSubmit(onSubmit)}
            loading={isSubmitting}
            disabled={isButtonDisabled}
          >
            {mode === 'create' ? (
              <FormattedMessage defaultMessage="Create" description="Create button" />
            ) : (
              <FormattedMessage defaultMessage="Save changes" description="Save changes button" />
            )}
          </Button>
        </Tooltip>
      </div>
    </>
  );
};

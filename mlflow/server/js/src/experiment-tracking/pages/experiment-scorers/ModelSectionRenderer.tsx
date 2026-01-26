import React from 'react';
import type { Control, UseFormSetValue } from 'react-hook-form';
import { Controller, useWatch } from 'react-hook-form';
import { useDesignSystemTheme, Typography, Input, FormUI } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { COMPONENT_ID_PREFIX, type ScorerFormMode, SCORER_FORM_MODE } from './constants';
import { EndpointSelector } from '../../components/EndpointSelector';
import {
  ModelProvider,
  getModelProvider,
  getEndpointNameFromGatewayModel,
  formatGatewayModelFromEndpoint,
} from '../../../gateway/utils/gatewayUtils';
import type { LLMScorerFormData } from './LLMScorerFormRenderer';

export interface ModelSectionRendererProps {
  mode: ScorerFormMode;
  control: Control<LLMScorerFormData>;
  setValue: UseFormSetValue<LLMScorerFormData>;
  onUserSelect?: (fieldName: keyof LLMScorerFormData, value: string) => void;
}

export const ModelSectionRenderer: React.FC<ModelSectionRendererProps> = ({
  mode,
  control,
  setValue,
  onUserSelect,
}) => {
  const { theme } = useDesignSystemTheme();

  const currentModel = useWatch({ control, name: 'model' });
  const modelInputMode = useWatch({ control, name: 'modelInputMode' });
  const currentEndpointName = getEndpointNameFromGatewayModel(currentModel);

  // In DISPLAY mode, always derive from the actual model value since it's read-only.
  // In CREATE/EDIT mode, use the form field so users can toggle between input modes while editing.
  const modelProvider =
    mode === SCORER_FORM_MODE.DISPLAY ? getModelProvider(currentModel) : (modelInputMode ?? ModelProvider.GATEWAY);

  const handleSwitchProvider = (targetProvider: ModelProvider) => {
    setValue('modelInputMode', targetProvider);
    setValue('model', '', { shouldValidate: true });
    // Toggle automatic evaluation based on model provider:
    // - Disable when switching to non-gateway model (automatic evaluation only works with gateway)
    // - Re-enable when switching back to gateway model
    if (targetProvider === ModelProvider.OTHER) {
      setValue('sampleRate', 0);
    } else if (targetProvider === ModelProvider.GATEWAY) {
      setValue('sampleRate', 100);
    }
  };

  const stopPropagationClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const isReadOnly = mode === SCORER_FORM_MODE.DISPLAY;

  if (modelProvider === ModelProvider.OTHER) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor="mlflow-experiment-scorers-model" required>
          <FormattedMessage defaultMessage="Model" description="Section header for model selection" />
        </FormUI.Label>
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="Enter a model identifier (e.g., openai:/gpt-4.1-mini). Scorers using direct models must configure API keys in your local environment."
            description="Hint text for direct model input"
          />
        </FormUI.Hint>
        <Controller
          name="model"
          control={control}
          rules={{ required: true }}
          render={({ field }) => (
            <Input
              {...field}
              componentId={`${COMPONENT_ID_PREFIX}.model-input`}
              id="mlflow-experiment-scorers-model"
              disabled={isReadOnly}
              placeholder="openai:/gpt-4.1-mini"
              css={{ marginTop: theme.spacing.sm }}
              onClick={stopPropagationClick}
            />
          )}
        />
        {!isReadOnly && (
          <div css={{ marginTop: theme.spacing.sm }}>
            <Typography.Link
              componentId={`${COMPONENT_ID_PREFIX}.switch-to-endpoint-link`}
              onClick={() => handleSwitchProvider(ModelProvider.GATEWAY)}
              css={{ cursor: 'pointer' }}
            >
              <FormattedMessage
                defaultMessage="← Use an endpoint instead"
                description="Link to switch from direct model to endpoint selection"
              />
            </Typography.Link>
          </div>
        )}
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <FormUI.Label htmlFor="mlflow-experiment-scorers-model" required>
        <FormattedMessage defaultMessage="Model" description="Section header for model selection" />
      </FormUI.Label>
      <FormUI.Hint>
        <FormattedMessage
          defaultMessage="Select an endpoint to use for this judge."
          description="Hint text for endpoint selection"
        />
      </FormUI.Hint>
      <Controller
        name="model"
        control={control}
        rules={{ required: true }}
        render={({ field }) => (
          <div css={{ marginTop: theme.spacing.sm }} onClick={stopPropagationClick}>
            <EndpointSelector
              currentEndpointName={currentEndpointName}
              onEndpointSelect={(endpointName) => {
                const modelValue = formatGatewayModelFromEndpoint(endpointName);
                setValue('model', modelValue, { shouldValidate: true });
                onUserSelect?.('model', modelValue);
              }}
              disabled={isReadOnly}
              componentIdPrefix={`${COMPONENT_ID_PREFIX}.endpoint`}
            />
          </div>
        )}
      />
      {!isReadOnly && (
        <div css={{ marginTop: theme.spacing.sm }}>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage
              defaultMessage="Or {enterManually}"
              description="Text with link to switch to direct model identifier input"
              values={{
                enterManually: (
                  <Typography.Link
                    componentId={`${COMPONENT_ID_PREFIX}.switch-to-manual-link`}
                    onClick={() => handleSwitchProvider(ModelProvider.OTHER)}
                    css={{ cursor: 'pointer' }}
                  >
                    <FormattedMessage
                      defaultMessage="enter a model identifier"
                      description="Link text to switch to direct model identifier input"
                    />
                  </Typography.Link>
                ),
              }}
            />
          </Typography.Text>
        </div>
      )}
    </div>
  );
};

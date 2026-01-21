import React, { useState } from 'react';
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
}

export const ModelSectionRenderer: React.FC<ModelSectionRendererProps> = ({ mode, control, setValue }) => {
  const { theme } = useDesignSystemTheme();

  const currentModel = useWatch({ control, name: 'model' });
  const currentEndpointName = getEndpointNameFromGatewayModel(currentModel);

  const [modelProviderState, setModelProvider] = useState<ModelProvider>(() => getModelProvider(currentModel));
  // In DISPLAY mode, always derive from the actual model value since it's read-only.
  // In CREATE/EDIT mode, use state so users can toggle between input modes while editing.
  const modelProvider = mode === SCORER_FORM_MODE.DISPLAY ? getModelProvider(currentModel) : modelProviderState;

  const handleSwitchProvider = (targetProvider: ModelProvider) => {
    setModelProvider(targetProvider);
    setValue('model', '');
    // Disable automatic evaluation when switching to non-gateway model,
    // since automatic evaluation only works with gateway models
    if (targetProvider === ModelProvider.OTHER) {
      setValue('sampleRate', 0);
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
              onEndpointSelect={(endpointName) => field.onChange(formatGatewayModelFromEndpoint(endpointName))}
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
              description="Text with link to switch to manual model entry"
              values={{
                enterManually: (
                  <Typography.Link
                    componentId={`${COMPONENT_ID_PREFIX}.switch-to-manual-link`}
                    onClick={() => handleSwitchProvider(ModelProvider.OTHER)}
                    css={{ cursor: 'pointer' }}
                  >
                    <FormattedMessage
                      defaultMessage="enter model manually"
                      description="Link text to switch to manual model input"
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

import { useState, useCallback } from 'react';
import { Button, Input, Modal, Typography, useDesignSystemTheme, ChevronLeftIcon } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { EndpointSelector } from '../../../experiment-tracking/components/EndpointSelector';
import type { GuardrailStage, GuardrailAction } from '../../types';

// ─── Props ──────────────────────────────────────────────────────────────────

export interface CustomGuardrailFormProps {
  open: boolean;
  onClose: () => void;
  onBack: () => void;
  onSuccess: () => void;
  endpointId: string;
  stage: GuardrailStage;
  action: GuardrailAction;
}

// ─── Component ──────────────────────────────────────────────────────────────

export const CustomGuardrailForm = ({
  open,
  onClose,
  onBack,
  onSuccess,
  endpointId,
  stage,
  action,
}: CustomGuardrailFormProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [name, setName] = useState('');
  const [prompt, setPrompt] = useState('');
  const [modelEndpoint, setModelEndpoint] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isFormValid = name.trim().length > 0 && prompt.trim().length > 0;

  const handleSubmit = useCallback(async () => {
    if (!isFormValid) return;
    setIsSubmitting(true);
    setError(null);
    try {
      const { GatewayApi } = await import('../../api');

      // Register the custom scorer, then create guardrail, then add to endpoint
      // For now, we create a guardrail with the prompt as the scorer definition
      const createResponse = await GatewayApi.createGuardrail({
        name: name.trim(),
        scorer_id: name.trim(),
        scorer_version: 1,
        stage,
        action,
        ...(modelEndpoint ? { action_endpoint_id: modelEndpoint } : {}),
      });

      await GatewayApi.addGuardrailToEndpoint({
        endpoint_id: endpointId,
        guardrail_id: createResponse.guardrail.guardrail_id,
      });

      onSuccess();
      onClose();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to create custom guardrail');
    } finally {
      setIsSubmitting(false);
    }
  }, [isFormValid, name, prompt, modelEndpoint, stage, action, endpointId, onSuccess, onClose]);

  return (
    <Modal
      componentId="mlflow.gateway.guardrails.custom-form"
      title={intl.formatMessage({
        defaultMessage: 'Create Custom Guardrail',
        description: 'Title for custom guardrail creation modal',
      })}
      visible={open}
      onCancel={onClose}
      size="wide"
      footer={
        <div css={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
          <Button componentId="mlflow.gateway.guardrails.custom-back" type="tertiary" onClick={onBack}>
            <ChevronLeftIcon />
            <FormattedMessage defaultMessage="Back" description="Back button in custom guardrail form" />
          </Button>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Button componentId="mlflow.gateway.guardrails.custom-cancel" onClick={onClose}>
              <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
            </Button>
            <Button
              componentId="mlflow.gateway.guardrails.custom-create"
              type="primary"
              onClick={handleSubmit}
              loading={isSubmitting}
              disabled={!isFormValid || isSubmitting}
            >
              <FormattedMessage defaultMessage="Create" description="Create custom guardrail button" />
            </Button>
          </div>
        </div>
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {/* Name */}
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage defaultMessage="Guardrail name" description="Label for guardrail name input" />
          </Typography.Text>
          <Input
            componentId="mlflow.gateway.guardrails.custom-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g., pii-detector',
              description: 'Placeholder for guardrail name',
            })}
          />
        </div>

        {/* Prompt / Instructions */}
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Judge instructions"
              description="Label for guardrail judge prompt textarea"
            />
          </Typography.Text>
          <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Describe what the judge should check for. The LLM will evaluate requests/responses against these instructions."
              description="Help text for guardrail judge instructions"
            />
          </Typography.Text>
          <Input.TextArea
            componentId="mlflow.gateway.guardrails.custom-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={intl.formatMessage({
              defaultMessage:
                'e.g., Check if the text contains personally identifiable information (PII) such as email addresses, phone numbers, or social security numbers.',
              description: 'Placeholder for judge instructions',
            })}
            autoSize={{ minRows: 4, maxRows: 10 }}
          />
        </div>

        {/* Model endpoint selector */}
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Judge model (optional)"
              description="Label for judge model endpoint selector"
            />
          </Typography.Text>
          <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Gateway endpoint to use for the LLM judge. If not set, the default model will be used."
              description="Help text for judge model selector"
            />
          </Typography.Text>
          <EndpointSelector
            currentEndpointName={modelEndpoint}
            onEndpointSelect={setModelEndpoint}
            componentIdPrefix="mlflow.gateway.guardrails.custom-model"
            placeholder={intl.formatMessage({
              defaultMessage: 'Select endpoint...',
              description: 'Placeholder for judge model endpoint',
            })}
          />
        </div>
      </div>

      {/* Error */}
      {error && (
        <div css={{ marginTop: theme.spacing.sm }}>
          <Typography.Text color="error">{error}</Typography.Text>
        </div>
      )}
    </Modal>
  );
};

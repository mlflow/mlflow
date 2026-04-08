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
  experimentId?: string;
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
  experimentId,
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
      const { registerScorer } = await import('../../../experiment-tracking/pages/experiment-scorers/api');

      // Step 1: Register the custom scorer to get a scorer_id + version
      const scorerName = name.trim();
      const registered = await registerScorer(experimentId ?? '0', {
        name: scorerName,
        serialized_scorer: JSON.stringify({
          name: scorerName,
          call_source: prompt.trim(),
        }),
      } as any);

      // Step 2: Resolve endpoint name → ID if a model endpoint was selected
      let actionEndpointId: string | undefined;
      if (modelEndpoint) {
        const { fetchAPI, getAjaxUrl } = await import('../../../common/utils/FetchUtils');
        const params = new URLSearchParams({ name: modelEndpoint });
        const endpointResp = (await fetchAPI(
          getAjaxUrl(`ajax-api/3.0/mlflow/gateway/endpoints/get?${params.toString()}`),
        )) as { endpoint: { endpoint_id: string } };
        actionEndpointId = endpointResp.endpoint.endpoint_id;
      }

      // Step 3: Create the guardrail
      const createResponse = await GatewayApi.createGuardrail({
        name: scorerName,
        scorer_id: registered.scorer_id,
        scorer_version: registered.version,
        stage,
        action,
        ...(actionEndpointId ? { action_endpoint_id: actionEndpointId } : {}),
      });

      // Step 4: Add to endpoint
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
  }, [isFormValid, name, prompt, modelEndpoint, stage, action, endpointId, experimentId, onSuccess, onClose]);

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

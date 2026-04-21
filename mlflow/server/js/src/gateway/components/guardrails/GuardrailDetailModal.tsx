import { useState, useCallback, useEffect } from 'react';
import { Button, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { GatewayGuardrailConfig, GuardrailStage, GuardrailAction } from '../../types';
import { GatewayApi } from '../../api';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import { registerScorer } from '../../../experiment-tracking/pages/experiment-scorers/api';
import { TEMPLATE_INSTRUCTIONS_MAP } from '../../../experiment-tracking/pages/experiment-scorers/prompts';
import { EndpointSelector } from '../../../experiment-tracking/components/EndpointSelector';
import { STAGE_HINTS, validateStageInstructions } from './guardrailValidation';
import { PipelineStagePicker } from './PipelineStagePicker';
import { ActionPicker } from './ActionPicker';

// ─── Props ──────────────────────────────────────────────────────────────────

export interface GuardrailDetailModalProps {
  open: boolean;
  onClose: () => void;
  onDelete: (guardrailId: string) => void;
  onSuccess: () => void;
  endpointId: string;
  experimentId?: string;
  guardrailConfig: GatewayGuardrailConfig | null;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function parseScorer(serializedScorer: string | undefined) {
  try {
    if (serializedScorer) {
      return JSON.parse(serializedScorer);
    }
  } catch {
    // ignore parse errors
  }
  return null;
}

function extractInitialPrompt(parsedScorer: any): string {
  if (parsedScorer?.instructions_judge_pydantic_data?.instructions) {
    return parsedScorer.instructions_judge_pydantic_data.instructions;
  }
  // Legacy fallbacks for older stored formats
  if (parsedScorer?.builtin_scorer_pydantic_data?.instructions) {
    return parsedScorer.builtin_scorer_pydantic_data.instructions;
  }
  if (parsedScorer?.instructions) return parsedScorer.instructions;
  if (parsedScorer?.call_source) return parsedScorer.call_source;
  if (parsedScorer?.builtin_scorer_class) {
    return TEMPLATE_INSTRUCTIONS_MAP[parsedScorer.builtin_scorer_class] ?? parsedScorer.builtin_scorer_class;
  }
  return '';
}

function extractInitialModelEndpoint(
  parsedScorer: any,
  endpoints: { name: string; endpoint_id: string }[],
): string | undefined {
  const model = parsedScorer?.instructions_judge_pydantic_data?.model as string | undefined;
  if (!model?.startsWith('gateway:/')) return undefined;
  const value = model.slice('gateway:/'.length);
  const endpoint = endpoints.find((e) => e.name === value) ?? endpoints.find((e) => e.endpoint_id === value);
  return endpoint?.name ?? value;
}

// ─── Modal ──────────────────────────────────────────────────────────────────

export const GuardrailDetailModal = ({
  open,
  onClose,
  onDelete,
  onSuccess,
  endpointId,
  experimentId,
  guardrailConfig,
}: GuardrailDetailModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { data: endpoints = [] } = useEndpointsQuery();

  const guardrail = guardrailConfig?.guardrail;
  const [stage, setStage] = useState<GuardrailStage>(guardrail?.stage ?? 'BEFORE');
  const [action, setAction] = useState<GuardrailAction>(guardrail?.action ?? 'VALIDATION');
  const [prompt, setPrompt] = useState('');
  const [modelEndpoint, setModelEndpoint] = useState<string | undefined>(undefined);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const parsedScorer = parseScorer(guardrail?.scorer?.serialized_scorer);
  const initialPrompt = extractInitialPrompt(parsedScorer);
  const initialModelEndpoint = extractInitialModelEndpoint(parsedScorer, endpoints);

  // Reset form when guardrail changes
  useEffect(() => {
    if (open && guardrail) {
      setStage(guardrail.stage);
      setAction(guardrail.action);
      setPrompt(initialPrompt);
      setModelEndpoint(initialModelEndpoint);
      setError(null);
    }
  }, [open, guardrail, initialPrompt, initialModelEndpoint, endpoints]);

  const hasChanges =
    guardrail &&
    (stage !== guardrail.stage ||
      action !== guardrail.action ||
      prompt !== initialPrompt ||
      modelEndpoint !== initialModelEndpoint);

  const instructionsError = validateStageInstructions(prompt, stage);

  const handleStageChange = useCallback((newStage: GuardrailStage) => {
    setPrompt((prev) => {
      if (newStage === 'AFTER') {
        return prev.replace(/\{\{\s*inputs\s*\}\}/g, '{{ outputs }}');
      } else {
        return prev.replace(/\{\{\s*outputs\s*\}\}/g, '{{ inputs }}');
      }
    });
    setStage(newStage);
  }, []);

  const handleSave = useCallback(async () => {
    if (!guardrailConfig || !guardrail || !hasChanges) return;
    setIsSaving(true);
    setError(null);
    try {
      let scorerId = guardrail.scorer?.scorer_id;
      let scorerVersion = guardrail.scorer?.scorer_version;

      // If prompt or judge model changed, register a new scorer version
      if (prompt !== initialPrompt || modelEndpoint !== initialModelEndpoint) {
        const scorerName = parsedScorer?.name ?? guardrail.name;
        const newScorer = await registerScorer(experimentId ?? '0', {
          name: scorerName,
          serialized_scorer: JSON.stringify({
            name: scorerName,
            instructions_judge_pydantic_data: {
              ...parsedScorer?.instructions_judge_pydantic_data,
              instructions: prompt,
              feedback_value_type: { type: 'string', enum: ['yes', 'no'] },
              ...(modelEndpoint ? { model: `gateway:/${modelEndpoint}` } : { model: undefined }),
            },
          }),
        } as any);
        scorerId = newScorer.scorer_id;
        scorerVersion = newScorer.version;
      }

      if (!scorerId || scorerVersion === undefined) {
        throw new Error('Cannot update guardrail: scorer information is missing');
      }

      // Create new guardrail first, then unlink old — so a creation failure
      // doesn't leave the endpoint without a guardrail.
      const newGuardrail = await GatewayApi.createGuardrail({
        name: guardrail.name,
        scorer_id: scorerId,
        scorer_version: scorerVersion,
        stage,
        action,
        ...(action === 'SANITIZATION' && modelEndpoint
          ? { action_endpoint_id: endpoints.find((e) => e.name === modelEndpoint)?.endpoint_id }
          : {}),
      });

      await GatewayApi.addGuardrailToEndpoint({
        endpoint_id: endpointId,
        guardrail_id: newGuardrail.guardrail.guardrail_id,
        execution_order: guardrailConfig.execution_order,
      });

      await GatewayApi.removeGuardrailFromEndpoint({
        endpoint_id: endpointId,
        guardrail_id: guardrailConfig.guardrail_id,
      });

      onSuccess();
      onClose();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to update guardrail');
    } finally {
      setIsSaving(false);
    }
  }, [
    guardrailConfig,
    guardrail,
    hasChanges,
    prompt,
    initialPrompt,
    modelEndpoint,
    initialModelEndpoint,
    endpoints,
    parsedScorer,
    stage,
    action,
    endpointId,
    experimentId,
    onSuccess,
    onClose,
  ]);

  const handleDelete = useCallback(() => {
    if (!guardrailConfig) return;
    onDelete(guardrailConfig.guardrail_id);
    onClose();
  }, [guardrailConfig, onDelete, onClose]);

  if (!guardrailConfig) return null;

  return (
    <Modal
      componentId="mlflow.gateway.guardrails.detail-modal"
      title={intl.formatMessage(
        { defaultMessage: 'Guardrail: {name}', description: 'Title for guardrail detail modal' },
        { name: guardrail?.name ?? guardrailConfig.guardrail_id },
      )}
      visible={open}
      onCancel={onClose}
      size="wide"
      footer={
        <div css={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
          <Button componentId="mlflow.gateway.guardrails.detail-delete" danger onClick={handleDelete}>
            <FormattedMessage defaultMessage="Delete" description="Delete guardrail button" />
          </Button>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Button componentId="mlflow.gateway.guardrails.detail-cancel" onClick={onClose}>
              <FormattedMessage defaultMessage="Close" description="Close button" />
            </Button>
            <Button
              componentId="mlflow.gateway.guardrails.detail-save"
              type="primary"
              onClick={handleSave}
              loading={isSaving}
              disabled={!hasChanges || isSaving || instructionsError !== null}
            >
              <FormattedMessage defaultMessage="Save" description="Save guardrail changes button" />
            </Button>
          </div>
        </div>
      }
    >
      <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
        <FormattedMessage
          defaultMessage="Review and edit the guardrail details, choose placement and action."
          description="Detail modal subtitle"
        />
      </Typography.Text>

      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        {/* Section: Guardrail Details */}
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div css={{ display: 'flex', alignItems: 'baseline', gap: theme.spacing.sm }}>
            <Typography.Text bold css={{ fontSize: theme.typography.fontSizeLg }}>
              <FormattedMessage defaultMessage="Guardrail Details" description="Edit modal section title" />
            </Typography.Text>
            {guardrail?.scorer && (
              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                v{guardrail.scorer.scorer_version}
              </Typography.Text>
            )}
          </div>
        </div>

        <PipelineStagePicker stage={stage} onChange={handleStageChange} />

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {/* Instructions */}
          <div>
            <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Instructions" description="Label for guardrail judge instructions" />
            </Typography.Text>
            <Typography.Text
              color="secondary"
              css={{ display: 'block', marginBottom: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
            >
              {STAGE_HINTS[stage]}
            </Typography.Text>
            <Input.TextArea
              componentId="mlflow.gateway.guardrails.detail-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              autoSize={{ minRows: 4, maxRows: 10 }}
            />
            {instructionsError && (
              <Typography.Text
                color="error"
                css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
              >
                {instructionsError}
              </Typography.Text>
            )}
          </div>

          {/* Guardrail model endpoint */}
          <div>
            <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Guardrail Model" description="Guardrail model label" />
            </Typography.Text>
            <EndpointSelector
              componentIdPrefix="mlflow.gateway.guardrails.detail-model"
              currentEndpointName={modelEndpoint}
              onEndpointSelect={setModelEndpoint}
              showCreateButton={false}
              excludeEndpointIds={[endpointId]}
            />
          </div>
        </div>

        <ActionPicker action={action} onActionChange={setAction} />

        {error && <Typography.Text color="error">{error}</Typography.Text>}
      </div>
    </Modal>
  );
};

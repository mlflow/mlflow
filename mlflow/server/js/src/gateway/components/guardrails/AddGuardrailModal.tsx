import { useState, useCallback, useEffect } from 'react';
import {
  Button,
  ChevronLeftIcon,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  GearIcon,
  Input,
  Modal,
  SparkleDoubleIcon,
  Tooltip,
  Typography,
  UserIcon,
  useDesignSystemEventComponentCallbacks,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useCreateGuardrail } from '../../hooks/useCreateGuardrail';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../../api';
import { GatewayQueryKeys } from '../../hooks/queryKeys';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import type { GatewayGuardrailConfig, GuardrailStage, GuardrailAction } from '../../types';
import { registerScorer } from '../../../experiment-tracking/pages/experiment-scorers/api';
import { TEMPLATE_INSTRUCTIONS_MAP } from '../../../experiment-tracking/pages/experiment-scorers/prompts';
import { LLM_TEMPLATE } from '../../../experiment-tracking/pages/experiment-scorers/types';
import { EndpointSelector } from '../../../experiment-tracking/components/EndpointSelector';
import { STAGE_HINTS, validateStageInstructions } from './guardrailValidation';
import { PipelineStagePicker } from './PipelineStagePicker';
import { ActionPicker } from './ActionPicker';

// ─── Guardrail type definitions ─────────────────────────────────────────────

interface GuardrailType {
  id: string;
  componentId: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  builtin: boolean;
  scorerClass?: LLM_TEMPLATE;
  presetInstructions?: string;
  defaultStage?: GuardrailStage;
}

const PII_PRESET_INSTRUCTIONS =
  'You are a PII (Personally Identifiable Information) detector. Analyze the given text for any personal ' +
  'data that could identify an individual, including names, email addresses, phone numbers, physical addresses, ' +
  'social security numbers, credit card numbers, dates of birth, IP addresses, or other identifying information. ' +
  'Your entire response must be a single, raw JSON object with no surrounding text or markdown.\n\n' +
  'The JSON object must be structured only using the following format. Do not use any markdown formatting ' +
  'or output additional lines.\n' +
  '{\n' +
  '    "rationale": "A concise explanation for your decision. Start each rationale with `Let\'s think step by step`",\n' +
  '    "result": "The string \'yes\' if the content contains no PII, or \'no\' if PII is detected."\n' +
  '}\n\n' +
  '<text>{{ inputs }}</text>';

const getGuardrailTypes = (intl: ReturnType<typeof useIntl>): GuardrailType[] => [
  {
    id: 'safety',
    componentId: 'mlflow.gateway.guardrails.type-card.safety',
    name: intl.formatMessage({ defaultMessage: 'Safety', description: 'Safety guardrail type name' }),
    description: intl.formatMessage({
      defaultMessage: 'Detects harmful, offensive, or toxic content in requests and responses.',
      description: 'Safety guardrail type description',
    }),
    icon: <SparkleDoubleIcon />,
    builtin: true,
    scorerClass: LLM_TEMPLATE.SAFETY,
    presetInstructions: TEMPLATE_INSTRUCTIONS_MAP[LLM_TEMPLATE.SAFETY],
    defaultStage: 'AFTER',
  },
  {
    id: 'pii',
    componentId: 'mlflow.gateway.guardrails.type-card.pii',
    name: intl.formatMessage({ defaultMessage: 'PII Detection', description: 'PII guardrail type name' }),
    description: intl.formatMessage({
      defaultMessage: 'Detects personally identifiable information such as names, emails, and phone numbers.',
      description: 'PII guardrail type description',
    }),
    icon: <UserIcon />,
    builtin: true,
    presetInstructions: PII_PRESET_INSTRUCTIONS,
    defaultStage: 'BEFORE',
  },
  {
    id: 'custom',
    componentId: 'mlflow.gateway.guardrails.type-card.custom',
    name: intl.formatMessage({ defaultMessage: 'Custom Guardrail', description: 'Custom guardrail type name' }),
    description: intl.formatMessage({
      defaultMessage: 'Define your own guardrail with a custom name and instructions.',
      description: 'Custom guardrail type description',
    }),
    icon: <GearIcon />,
    builtin: false,
  },
];

// ─── Type card ──────────────────────────────────────────────────────────────

const TypeCard = ({
  componentId,
  type,
  onClick,
}: {
  componentId: string;
  type: GuardrailType;
  onClick: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId,
    analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
  });

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      eventContext.onClick(e);
      onClick();
    },
    [eventContext, onClick],
  );

  return (
    <div
      role="option"
      aria-selected={false}
      onClick={handleClick}
      onKeyDown={(e) => e.key === 'Enter' && onClick()}
      tabIndex={0}
      css={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
        borderRadius: theme.borders.borderRadiusMd,
        border: `2px solid ${theme.colors.border}`,
        cursor: 'pointer',
        '&:hover': { borderColor: theme.colors.actionPrimaryBackgroundDefault },
        transition: 'border-color 0.15s',
      }}
    >
      <div css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeLg, flexShrink: 0 }}>
        {type.icon}
      </div>
      <div css={{ flex: 1, minWidth: 0 }}>
        <Typography.Text bold>{type.name}</Typography.Text>
        <Typography.Text color="secondary" css={{ display: 'block', fontSize: theme.typography.fontSizeSm }}>
          {type.description}
        </Typography.Text>
      </div>
    </div>
  );
};

// ─── Props ──────────────────────────────────────────────────────────────────

export interface GuardrailModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  onDelete?: (guardrailId: string) => void;
  endpointName: string;
  endpointId: string;
  editingGuardrail?: GatewayGuardrailConfig | null;
  experimentId?: string;
}

// ─── Modal ──────────────────────────────────────────────────────────────────

export const AddGuardrailModal = ({ open, onClose, onSuccess, endpointId, experimentId }: GuardrailModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const guardrailTypes = getGuardrailTypes(intl);
  const { mutateAsync: createGuardrail } = useCreateGuardrail();
  const queryClient = useQueryClient();
  const { data: endpoints = [], isLoading: isEndpointsLoading, error: endpointsError } = useEndpointsQuery();

  const [step, setStep] = useState<1 | 2>(1);
  const [name, setName] = useState('');
  const [instructions, setInstructions] = useState('');
  const [modelEndpoint, setModelEndpoint] = useState<string | undefined>(undefined);
  const [stage, setStage] = useState<GuardrailStage>('BEFORE');
  const [action, setAction] = useState<GuardrailAction>('VALIDATION');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setStep(1);
      setName('');
      setInstructions('');
      setModelEndpoint(undefined);
      setStage('BEFORE');
      setAction('VALIDATION');
      setError(null);
    }
  }, [open]);

  const handleStageChange = useCallback((newStage: GuardrailStage) => {
    setInstructions((prev) => {
      if (newStage === 'AFTER') {
        return prev.replace(/\{\{\s*inputs\s*\}\}/g, '{{ outputs }}');
      } else {
        return prev.replace(/\{\{\s*outputs\s*\}\}/g, '{{ inputs }}');
      }
    });
    setStage(newStage);
  }, []);

  const handleSelectType = useCallback((type: GuardrailType) => {
    if (type.builtin) {
      setName(type.name);
      setInstructions(type.presetInstructions ?? '');
      setStage(type.defaultStage ?? 'BEFORE');
    } else {
      setName('');
      setInstructions('');
    }
    setStep(2);
  }, []);

  const handleBack = useCallback(() => {
    setStep(1);
    setError(null);
  }, []);

  const handleCreate = useCallback(async () => {
    if (!name.trim() || !modelEndpoint) return;
    setIsSubmitting(true);
    setError(null);
    try {
      const selectedModelEndpoint = endpoints.find((endpoint) => endpoint.name === modelEndpoint);
      if (!selectedModelEndpoint) {
        throw new Error(
          intl.formatMessage({
            defaultMessage: 'Selected guardrail model endpoint is unavailable. Please choose another endpoint.',
            description: 'Error shown when selected guardrail model endpoint is no longer available',
          }),
        );
      }

      const scorerName = name.trim().toLowerCase();
      const trimmedInstructions = instructions.trim();
      const serializedScorer = {
        name: scorerName,
        instructions_judge_pydantic_data: {
          instructions: trimmedInstructions,
          feedback_value_type: { type: 'string', enum: ['yes', 'no'] },
          model: `gateway:/${modelEndpoint}`,
        },
      };

      const registered = await registerScorer(experimentId ?? '0', {
        name: scorerName,
        serialized_scorer: JSON.stringify(serializedScorer),
      } as any);

      const createResponse = await createGuardrail({
        name: name.trim(),
        scorer_id: registered.scorer_id,
        scorer_version: registered.version,
        stage,
        action,
        ...(action === 'SANITIZATION' ? { action_endpoint_id: selectedModelEndpoint.endpoint_id } : {}),
      });

      await GatewayApi.addGuardrailToEndpoint({
        endpoint_id: endpointId,
        guardrail_id: createResponse.guardrail.guardrail_id,
      });
      queryClient.invalidateQueries(GatewayQueryKeys.guardrails);

      onSuccess();
      onClose();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to create guardrail');
    } finally {
      setIsSubmitting(false);
    }
  }, [
    name,
    instructions,
    modelEndpoint,
    endpoints,
    stage,
    action,
    endpointId,
    experimentId,
    createGuardrail,
    intl,
    onSuccess,
    onClose,
    queryClient,
  ]);

  const instructionsError = validateStageInstructions(instructions, stage);
  const isStep2Valid = name.trim().length > 0 && instructionsError === null;
  const endpointsLoaded = !isEndpointsLoading && !endpointsError;
  const hasAvailableGuardrailModelEndpoint =
    !endpointsLoaded || endpoints.some((endpoint) => endpoint.endpoint_id !== endpointId);
  const createButtonTooltip = !hasAvailableGuardrailModelEndpoint
    ? intl.formatMessage({
        defaultMessage: 'You need another endpoint to use guardrails.',
        description: 'Tooltip shown when no alternate endpoint exists for guardrail model selection',
      })
    : !modelEndpoint
      ? intl.formatMessage({
          defaultMessage: 'Select a Guardrail Model endpoint to create this guardrail.',
          description: 'Tooltip shown when create button is disabled because guardrail model is not selected',
        })
      : undefined;
  const isCreateButtonDisabled = !isStep2Valid || isSubmitting || !modelEndpoint;

  return (
    <Modal
      componentId="mlflow.gateway.guardrails.add-modal"
      title={intl.formatMessage({
        defaultMessage: 'Create Guardrail',
        description: 'Title for create guardrail modal',
      })}
      visible={open}
      onCancel={onClose}
      size="wide"
      footer={
        step === 1 ? (
          <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button componentId="mlflow.gateway.guardrails.cancel" onClick={onClose}>
              <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
            </Button>
          </div>
        ) : (
          <div css={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
            <Button
              componentId="mlflow.gateway.guardrails.back"
              type="tertiary"
              icon={<ChevronLeftIcon />}
              onClick={handleBack}
            >
              <FormattedMessage defaultMessage="Back" description="Back button" />
            </Button>
            <Tooltip componentId="mlflow.gateway.guardrails.create-tooltip" content={createButtonTooltip}>
              <Button
                componentId="mlflow.gateway.guardrails.create"
                type="primary"
                onClick={handleCreate}
                loading={isSubmitting}
                disabled={isCreateButtonDisabled}
              >
                <FormattedMessage defaultMessage="Create Guardrail" description="Create guardrail button" />
              </Button>
            </Tooltip>
          </div>
        )
      }
    >
      {step === 1 && (
        <>
          <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
            <FormattedMessage
              defaultMessage="Select the type of guardrail you want to create."
              description="Step 1 subtitle"
            />
          </Typography.Text>
          <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.md }}>
            {guardrailTypes.map((type) => (
              <TypeCard
                key={type.id}
                componentId={type.componentId}
                type={type}
                onClick={() => handleSelectType(type)}
              />
            ))}
          </div>
        </>
      )}

      {step === 2 && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <div>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="Name" description="Guardrail name label" />
              </Typography.Text>
              <Input
                componentId="mlflow.gateway.guardrails.config-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={intl.formatMessage({
                  defaultMessage: 'e.g., PII Detection & Redaction',
                  description: 'Guardrail name placeholder',
                })}
              />
            </div>
          </div>

          <PipelineStagePicker stage={stage} onChange={handleStageChange} />

          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <div>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="Instructions" description="Guardrail instructions label" />
              </Typography.Text>
              <Typography.Text
                color="secondary"
                css={{
                  display: 'block',
                  marginBottom: theme.spacing.xs,
                  fontSize: theme.typography.fontSizeSm,
                  whiteSpace: 'pre-line',
                }}
              >
                {STAGE_HINTS[stage]}
              </Typography.Text>
              <Input.TextArea
                componentId="mlflow.gateway.guardrails.config-instructions"
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Describe what this guardrail should check for...',
                  description: 'Guardrail instructions placeholder',
                })}
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

            <div>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="Guardrail Model" description="Guardrail model label" />
              </Typography.Text>
              <EndpointSelector
                componentIdPrefix="mlflow.gateway.guardrails.config-model"
                currentEndpointName={modelEndpoint}
                onEndpointSelect={setModelEndpoint}
                disabled={!hasAvailableGuardrailModelEndpoint}
                showCreateButton={false}
                excludeEndpointIds={[endpointId]}
              />
              {!hasAvailableGuardrailModelEndpoint && (
                <Typography.Text
                  color="secondary"
                  css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
                >
                  <FormattedMessage
                    defaultMessage="You need another endpoint to use guardrails."
                    description="Guidance shown when no alternate endpoint exists for guardrail model selection"
                  />
                </Typography.Text>
              )}
            </div>
          </div>

          <ActionPicker action={action} onActionChange={setAction} />

          {error && <Typography.Text color="error">{error}</Typography.Text>}
        </div>
      )}
    </Modal>
  );
};

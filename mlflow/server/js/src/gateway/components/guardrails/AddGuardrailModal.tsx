import { useState, useCallback, useEffect } from 'react';
import {
  Button,
  Input,
  Modal,
  NoIcon,
  SparkleIcon,
  Typography,
  useDesignSystemTheme,
  SparkleDoubleIcon,
  GearIcon,
  ChevronLeftIcon,
  CheckCircleIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useCreateGuardrail } from '../../hooks/useCreateGuardrail';
import { GatewayApi } from '../../api';
import type { GatewayGuardrailConfig, GuardrailStage, GuardrailAction } from '../../types';
import { TEMPLATE_INSTRUCTIONS_MAP } from '../../../experiment-tracking/pages/experiment-scorers/prompts';
import { LLM_TEMPLATE } from '../../../experiment-tracking/pages/experiment-scorers/types';

// ─── Guardrail type definitions ─────────────────────────────────────────────

interface GuardrailType {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  builtin: boolean;
}

const GUARDRAIL_TYPES: GuardrailType[] = [
  {
    id: 'safety',
    name: 'Safety',
    description: 'Detects harmful, offensive, or toxic content in requests and responses.',
    icon: <SparkleDoubleIcon />,
    builtin: true,
  },
  {
    id: 'custom',
    name: 'Custom Guardrail',
    description: 'Define your own guardrail with a custom name and description.',
    icon: <GearIcon />,
    builtin: false,
  },
];

// ─── Shared select styles ───────────────────────────────────────────────────

const useSelectStyles = () => {
  const { theme } = useDesignSystemTheme();
  return {
    padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
    borderRadius: theme.borders.borderRadiusMd,
    border: `1px solid ${theme.colors.border}`,
    backgroundColor: theme.colors.backgroundPrimary,
    color: theme.colors.textPrimary,
    fontSize: theme.typography.fontSizeBase,
    cursor: 'pointer',
    width: '100%',
  } as const;
};

// ─── Type card ──────────────────────────────────────────────────────────────

const TypeCard = ({ type, selected, onClick }: { type: GuardrailType; selected: boolean; onClick: () => void }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="option"
      aria-selected={selected}
      onClick={onClick}
      onKeyDown={(e) => e.key === 'Enter' && onClick()}
      tabIndex={0}
      css={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
        borderRadius: theme.borders.borderRadiusMd,
        border: `2px solid ${selected ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.borderDecorative}`,
        cursor: 'pointer',
        backgroundColor: selected ? `${theme.colors.actionPrimaryBackgroundDefault}08` : 'transparent',
        '&:hover': {
          borderColor: theme.colors.actionPrimaryBackgroundDefault,
        },
        transition: 'border-color 0.15s, background-color 0.15s',
      }}
    >
      <div css={{ color: theme.colors.textSecondary, fontSize: 20, flexShrink: 0, marginTop: 2 }}>{type.icon}</div>
      <div css={{ flex: 1, minWidth: 0 }}>
        <Typography.Text bold>{type.name}</Typography.Text>
        <Typography.Text color="secondary" css={{ display: 'block', fontSize: theme.typography.fontSizeSm }}>
          {type.description}
        </Typography.Text>
      </div>
    </div>
  );
};

// ─── Step indicator ─────────────────────────────────────────────────────────

const StepIndicator = ({ step }: { step: 1 | 2 }) => {
  const { theme } = useDesignSystemTheme();
  const activeColor = theme.colors.actionPrimaryBackgroundDefault;
  const inactiveColor = theme.colors.textSecondary;

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.lg }}>
      {step === 2 ? (
        <CheckCircleIcon css={{ color: activeColor }} />
      ) : (
        <div
          css={{
            width: 24,
            height: 24,
            borderRadius: '50%',
            backgroundColor: activeColor,
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 12,
            fontWeight: 'bold',
          }}
        >
          1
        </div>
      )}
      <Typography.Text bold color={step >= 1 ? undefined : 'secondary'}>
        <FormattedMessage defaultMessage="Guardrail Type" description="Step 1 label" />
      </Typography.Text>
      <div css={{ flex: 1, height: 2, backgroundColor: step === 2 ? activeColor : theme.colors.borderDecorative }} />
      <div
        css={{
          width: 24,
          height: 24,
          borderRadius: '50%',
          backgroundColor: step === 2 ? activeColor : 'transparent',
          border: step === 2 ? 'none' : `2px solid ${inactiveColor}`,
          color: step === 2 ? '#fff' : inactiveColor,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 12,
          fontWeight: 'bold',
        }}
      >
        2
      </div>
      <Typography.Text bold color={step === 2 ? undefined : 'secondary'}>
        <FormattedMessage defaultMessage="Configuration" description="Step 2 label" />
      </Typography.Text>
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
  const selectStyles = useSelectStyles();
  const { mutateAsync: createGuardrail } = useCreateGuardrail();

  // Wizard state
  const [step, setStep] = useState<1 | 2>(1);
  const [selectedType, setSelectedType] = useState<string | null>(null);

  // Configuration state (step 2)
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [instructions, setInstructions] = useState('');
  const [stage, setStage] = useState<GuardrailStage>('BEFORE');
  const [action, setAction] = useState<GuardrailAction>('VALIDATION');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset on open
  useEffect(() => {
    if (open) {
      setStep(1);
      setSelectedType(null);
      setName('');
      setDescription('');
      setInstructions('');
      setStage('BEFORE');
      setAction('VALIDATION');
      setError(null);
    }
  }, [open]);

  // Pre-fill when moving to step 2
  const handleNext = useCallback(() => {
    if (!selectedType) return;
    const type = GUARDRAIL_TYPES.find((t) => t.id === selectedType);
    if (type?.builtin) {
      setName(type.name);
      setDescription(type.description);
      // Pre-fill with the builtin template prompt
      setInstructions(TEMPLATE_INSTRUCTIONS_MAP[LLM_TEMPLATE.SAFETY] ?? '');
    } else {
      setName('');
      setDescription('');
      setInstructions('');
    }
    setStep(2);
  }, [selectedType]);

  const handleBack = useCallback(() => {
    setStep(1);
    setError(null);
  }, []);

  // Submit
  const handleCreate = useCallback(async () => {
    if (!name.trim()) return;
    setIsSubmitting(true);
    setError(null);
    try {
      const { registerScorer } = await import('../../../experiment-tracking/pages/experiment-scorers/api');
      const type = GUARDRAIL_TYPES.find((t) => t.id === selectedType);

      // Register the scorer
      const scorerName = name.trim().toLowerCase();
      const serializedScorer = type?.builtin
        ? { name: scorerName, builtin_scorer_class: type.name, instructions: instructions.trim() }
        : { name: scorerName, call_source: instructions.trim(), instructions: instructions.trim() };

      const registered = await registerScorer(experimentId ?? '0', {
        name: scorerName,
        serialized_scorer: JSON.stringify(serializedScorer),
      } as any);

      // Create the guardrail
      const createResponse = await createGuardrail({
        name: name.trim(),
        scorer_id: registered.scorer_id,
        scorer_version: registered.version,
        stage,
        action,
      });

      // Add to endpoint
      await GatewayApi.addGuardrailToEndpoint({
        endpoint_id: endpointId,
        guardrail_id: createResponse.guardrail.guardrail_id,
      });

      onSuccess();
      onClose();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to create guardrail');
    } finally {
      setIsSubmitting(false);
    }
  }, [name, description, instructions, selectedType, stage, action, endpointId, experimentId, createGuardrail, onSuccess, onClose]);

  const isStep2Valid = name.trim().length > 0;

  return (
    <Modal
      componentId="mlflow.gateway.guardrails.add-modal"
      title={
        step === 1
          ? intl.formatMessage({ defaultMessage: 'Add Guardrail', description: 'Title for add guardrail modal step 1' })
          : intl.formatMessage({ defaultMessage: 'Configuration', description: 'Title for add guardrail modal step 2' })
      }
      visible={open}
      onCancel={onClose}
      size="wide"
      footer={
        step === 1 ? (
          <div css={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
            <Button componentId="mlflow.gateway.guardrails.cancel" onClick={onClose}>
              <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
            </Button>
            <Button
              componentId="mlflow.gateway.guardrails.next"
              type="primary"
              disabled={!selectedType}
              onClick={handleNext}
            >
              <FormattedMessage defaultMessage="Next" description="Next step button" />
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
            <Button
              componentId="mlflow.gateway.guardrails.create"
              type="primary"
              onClick={handleCreate}
              loading={isSubmitting}
              disabled={!isStep2Valid || isSubmitting}
            >
              <FormattedMessage defaultMessage="Create Guardrail" description="Create guardrail button" />
            </Button>
          </div>
        )
      }
    >
      {step === 2 && (
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
          <FormattedMessage
            defaultMessage="Review and edit the guardrail details, choose placement and action."
            description="Step 2 subtitle"
          />
        </Typography.Text>
      )}

      <StepIndicator step={step} />

      {step === 1 && (
        <>
          <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
            <FormattedMessage
              defaultMessage="Select the type of guardrail you want to add."
              description="Step 1 subtitle"
            />
          </Typography.Text>
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: theme.spacing.md,
            }}
          >
            {GUARDRAIL_TYPES.map((type) => (
              <TypeCard
                key={type.id}
                type={type}
                selected={selectedType === type.id}
                onClick={() => setSelectedType(type.id)}
              />
            ))}
          </div>
        </>
      )}

      {step === 2 && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {/* Section: Guardrail Details */}
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <Typography.Text bold css={{ fontSize: theme.typography.fontSizeLg }}>
              <FormattedMessage defaultMessage="Guardrail Details" description="Step 2 section title" />
            </Typography.Text>

            {/* Name */}
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

            {/* Description — judge instructions for all types */}
            <div>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="Description" description="Guardrail description label" />
              </Typography.Text>
              <Input.TextArea
                componentId="mlflow.gateway.guardrails.config-instructions"
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Enter judge instructions...',
                  description: 'Judge instructions placeholder',
                })}
                autoSize={{ minRows: 4, maxRows: 10 }}
              />
            </div>
          </div>

          {/* Section: Placement */}
          <div>
            <Typography.Text bold css={{ display: 'block', fontSize: theme.typography.fontSizeLg, marginBottom: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Placement" description="Guardrail placement label" />
            </Typography.Text>
            <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm, fontSize: theme.typography.fontSizeSm }}>
              <FormattedMessage
                defaultMessage="Click on a pipeline stage to choose where this guardrail runs."
                description="Placement help text"
              />
            </Typography.Text>
            {/* Pipeline visualizer */}
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: theme.spacing.sm,
                padding: `${theme.spacing.md}px ${theme.spacing.lg}px`,
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.borders.borderRadiusMd,
              }}
            >
              {(['Request', 'BEFORE', 'LLM', 'AFTER', 'Response'] as const).map((item, i) => {
                const isStage = item === 'BEFORE' || item === 'AFTER';
                const isSelected = isStage && item === stage;
                const label = item === 'BEFORE' ? 'Input Guardrails' : item === 'AFTER' ? 'Output Guardrails' : item;
                return (
                  <div key={item} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                    {i > 0 && (
                      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                        {'>'}
                      </Typography.Text>
                    )}
                    {isStage ? (
                      <div
                        role="option"
                        aria-selected={isSelected}
                        onClick={() => setStage(item as GuardrailStage)}
                        onKeyDown={(e) => e.key === 'Enter' && setStage(item as GuardrailStage)}
                        tabIndex={0}
                        css={{
                          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                          borderRadius: theme.borders.borderRadiusMd,
                          border: `1.5px dashed ${isSelected ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border}`,
                          cursor: 'pointer',
                          fontWeight: isSelected ? theme.typography.typographyBoldFontWeight : 'normal',
                          color: isSelected ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.textPrimary,
                          whiteSpace: 'nowrap',
                          userSelect: 'none',
                          '&:hover': { borderColor: theme.colors.actionPrimaryBackgroundDefault, color: theme.colors.actionPrimaryBackgroundDefault },
                        }}
                      >
                        {label}
                      </div>
                    ) : (
                      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, whiteSpace: 'nowrap' }}>
                        {label}
                      </Typography.Text>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Section: Action */}
          <div>
            <Typography.Text bold css={{ display: 'block', fontSize: theme.typography.fontSizeLg, marginBottom: theme.spacing.sm }}>
              <FormattedMessage defaultMessage="Action" description="Guardrail action label" />
            </Typography.Text>
            <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.md }}>
              {/* Block */}
              <div
                role="option"
                aria-selected={action === 'VALIDATION'}
                onClick={() => setAction('VALIDATION')}
                onKeyDown={(e) => e.key === 'Enter' && setAction('VALIDATION')}
                tabIndex={0}
                css={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: theme.spacing.sm,
                  padding: theme.spacing.md,
                  borderRadius: theme.borders.borderRadiusMd,
                  border: `2px solid ${action === 'VALIDATION' ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border}`,
                  cursor: 'pointer',
                  '&:hover': { borderColor: theme.colors.actionPrimaryBackgroundDefault },
                }}
              >
                <NoIcon css={{ fontSize: 18, color: theme.colors.textSecondary, flexShrink: 0, marginTop: 2 }} />
                <div>
                  <Typography.Text bold css={{ display: 'block' }}>
                    <FormattedMessage defaultMessage="Block" description="Block action title" />
                  </Typography.Text>
                  <Typography.Text color="secondary" css={{ display: 'block', fontSize: theme.typography.fontSizeSm }}>
                    <FormattedMessage
                      defaultMessage="Reject the request or response entirely and return an error."
                      description="Block action description"
                    />
                  </Typography.Text>
                </div>
              </div>
              {/* Sanitize */}
              <div
                role="option"
                aria-selected={action === 'SANITIZATION'}
                onClick={() => setAction('SANITIZATION')}
                onKeyDown={(e) => e.key === 'Enter' && setAction('SANITIZATION')}
                tabIndex={0}
                css={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: theme.spacing.sm,
                  padding: theme.spacing.md,
                  borderRadius: theme.borders.borderRadiusMd,
                  border: `2px solid ${action === 'SANITIZATION' ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border}`,
                  cursor: 'pointer',
                  '&:hover': { borderColor: theme.colors.actionPrimaryBackgroundDefault },
                }}
              >
                <SparkleIcon css={{ fontSize: 18, color: theme.colors.textSecondary, flexShrink: 0, marginTop: 2 }} />
                <div>
                  <Typography.Text bold css={{ display: 'block' }}>
                    <FormattedMessage defaultMessage="Sanitize" description="Sanitize action title" />
                  </Typography.Text>
                  <Typography.Text color="secondary" css={{ display: 'block', fontSize: theme.typography.fontSizeSm }}>
                    <FormattedMessage
                      defaultMessage="Redact or mask flagged content, then allow the request to continue."
                      description="Sanitize action description"
                    />
                  </Typography.Text>
                </div>
              </div>
            </div>
          </div>

          {error && (
            <div>
              <Typography.Text color="error">{error}</Typography.Text>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
};

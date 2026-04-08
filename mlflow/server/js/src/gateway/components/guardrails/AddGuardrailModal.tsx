import { useState, useMemo, useCallback, useEffect } from 'react';
import {
  Button,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  Typography,
  useDesignSystemTheme,
  SparkleDoubleIcon,
  GearIcon,
  SearchIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useAddGuardrail } from '../../hooks/useAddGuardrail';
import { GatewayApi } from '../../api';
import type { GatewayGuardrailConfig, GuardrailStage, GuardrailAction } from '../../types';

// ─── Builtin scorer definitions ─────────────────────────────────────────────

interface BuiltinScorer {
  name: string;
  description: string;
}

const BUILTIN_SCORERS: BuiltinScorer[] = [
  { name: 'Safety', description: "Does the app's response avoid harmful or toxic content?" },
];

// ─── Registered scorer item ─────────────────────────────────────────────────

interface RegisteredScorer {
  scorer_id: string;
  scorer_name: string;
  scorer_version: number;
  experiment_id: number;
}

// ─── Scorer row ─────────────────────────────────────────────────────────────

interface ScorerRowProps {
  icon: React.ReactNode;
  name: string;
  description: string;
  badge?: string;
  selected: boolean;
  onClick: () => void;
}

export const ScorerRow = ({ icon, name, description, badge, selected, onClick }: ScorerRowProps) => {
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
        alignItems: 'center',
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
        cursor: 'pointer',
        borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        backgroundColor: selected ? theme.colors.actionTertiaryBackgroundHover : 'transparent',
        '&:hover': {
          backgroundColor: theme.colors.actionTertiaryBackgroundHover,
        },
      }}
    >
      <div css={{ marginRight: theme.spacing.sm, color: theme.colors.textSecondary, display: 'flex' }}>{icon}</div>
      <div css={{ flex: 1, minWidth: 0 }}>
        <Typography.Text bold>{name}</Typography.Text>
        <Typography.Text color="secondary" css={{ display: 'block', fontSize: theme.typography.fontSizeSm }}>
          {description}
        </Typography.Text>
      </div>
      {badge && (
        <Typography.Text
          color="secondary"
          css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', flexShrink: 0 }}
        >
          {badge}
        </Typography.Text>
      )}
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
  const { mutateAsync: createGuardrail } = useAddGuardrail();

  const [stage, setStage] = useState<GuardrailStage>('BEFORE');
  const [action, setAction] = useState<GuardrailAction>('VALIDATION');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedScorer, setSelectedScorer] = useState<string | null>(null);
  const [registeredScorers, setRegisteredScorers] = useState<RegisteredScorer[]>([]);
  const [isLoadingScorers, setIsLoadingScorers] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch registered scorers when modal opens
  const fetchRegisteredScorers = useCallback(async () => {
    if (!experimentId) return;
    setIsLoadingScorers(true);
    try {
      const response = await fetch(
        `/ajax-api/3.0/mlflow/scorers/list?experiment_id=${encodeURIComponent(experimentId)}`,
      );
      if (response.ok) {
        const data = await response.json();
        setRegisteredScorers(data.scorers ?? []);
      }
    } catch {
      // Silently fail - registered scorers are optional
    } finally {
      setIsLoadingScorers(false);
    }
  }, [experimentId]);

  // Reset and fetch when modal opens
  useEffect(() => {
    if (open) {
      setSearchQuery('');
      setSelectedScorer(null);
      setError(null);
      fetchRegisteredScorers();
    }
  }, [open, fetchRegisteredScorers]);

  // Filter by search
  const filteredBuiltins = useMemo(() => {
    const q = searchQuery.toLowerCase();
    if (!q) return BUILTIN_SCORERS;
    return BUILTIN_SCORERS.filter((s) => s.name.toLowerCase().includes(q) || s.description.toLowerCase().includes(q));
  }, [searchQuery]);

  const filteredRegistered = useMemo(() => {
    const q = searchQuery.toLowerCase();
    if (!q) return registeredScorers;
    return registeredScorers.filter((s) => s.scorer_name.toLowerCase().includes(q));
  }, [searchQuery, registeredScorers]);

  // Submit: create guardrail then add to endpoint
  const handleAdd = useCallback(async () => {
    if (!selectedScorer) return;
    setIsSubmitting(true);
    setError(null);
    try {
      // Determine scorer_id and scorer_version
      const registered = registeredScorers.find((s) => s.scorer_id === selectedScorer);
      let scorerId: string;
      let scorerVersion: number;
      let guardrailName: string;

      if (registered) {
        scorerId = registered.scorer_id;
        scorerVersion = registered.scorer_version;
        guardrailName = registered.scorer_name;
      } else {
        // Builtin - the backend resolves the builtin scorer by name
        scorerId = selectedScorer;
        scorerVersion = 1;
        guardrailName = selectedScorer;
      }

      // Step 1: Create the guardrail
      const createResponse = await createGuardrail({
        name: guardrailName,
        scorer_id: scorerId,
        scorer_version: scorerVersion,
        stage,
        action,
      });

      // Step 2: Add to endpoint
      await GatewayApi.addGuardrailToEndpoint({
        endpoint_id: endpointId,
        guardrail_id: createResponse.guardrail.guardrail_id,
      });

      onSuccess();
      onClose();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to add guardrail');
    } finally {
      setIsSubmitting(false);
    }
  }, [selectedScorer, registeredScorers, stage, action, endpointId, createGuardrail, onSuccess, onClose]);

  return (
    <Modal
      componentId="mlflow.gateway.guardrails.add-modal"
      title={intl.formatMessage({ defaultMessage: 'Add Guardrail', description: 'Title for add guardrail modal' })}
      visible={open}
      onCancel={onClose}
      onOk={handleAdd}
      size="wide"
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.gateway.guardrails.cancel" onClick={onClose}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
          </Button>
          <Button
            componentId="mlflow.gateway.guardrails.add"
            type="primary"
            onClick={handleAdd}
            loading={isSubmitting}
            disabled={!selectedScorer || isSubmitting}
          >
            <FormattedMessage defaultMessage="Add" description="Add guardrail button" />
          </Button>
        </div>
      }
    >
      {/* Search */}
      <Input
        componentId="mlflow.gateway.guardrails.search"
        prefix={<SearchIcon />}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search guardrails...',
          description: 'Search placeholder in add guardrail modal',
        })}
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        css={{ marginBottom: theme.spacing.md }}
      />

      {/* Scorer list */}
      <div
        role="listbox"
        css={{
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          maxHeight: 400,
          overflowY: 'auto',
        }}
      >
        {isLoadingScorers && (
          <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
            <Spinner />
          </div>
        )}

        {/* Builtin scorers */}
        {filteredBuiltins.map((scorer) => (
          <ScorerRow
            key={scorer.name}
            icon={<SparkleDoubleIcon />}
            name={scorer.name}
            description={scorer.description}
            badge="BUILTIN"
            selected={selectedScorer === scorer.name}
            onClick={() => setSelectedScorer(scorer.name)}
          />
        ))}

        {/* Registered scorers */}
        {filteredRegistered.map((scorer) => (
          <ScorerRow
            key={scorer.scorer_id}
            icon={<GearIcon />}
            name={scorer.scorer_name}
            description={`v${scorer.scorer_version} \u00B7 experiment ${scorer.experiment_id}`}
            badge="REGISTERED"
            selected={selectedScorer === scorer.scorer_id}
            onClick={() => setSelectedScorer(scorer.scorer_id)}
          />
        ))}

        {!isLoadingScorers && filteredBuiltins.length === 0 && filteredRegistered.length === 0 && (
          <div css={{ padding: theme.spacing.lg, textAlign: 'center' }}>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="No scorers found matching your search."
                description="Empty state for scorer search in guardrail modal"
              />
            </Typography.Text>
          </div>
        )}
      </div>

      {/* Stage and Action selectors */}
      <div css={{ display: 'flex', gap: theme.spacing.lg, marginTop: theme.spacing.md }}>
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="When the guardrail runs"
              description="Label for guardrail stage selector"
            />
          </Typography.Text>
          <SimpleSelect
            id="guardrail-stage-select"
            componentId="mlflow.gateway.guardrails.stage-select"
            value={stage}
            onChange={({ target }) => {
              if (target.value !== stage) setStage(target.value as GuardrailStage);
            }}
          >
            <SimpleSelectOption value="BEFORE">Before LLM</SimpleSelectOption>
            <SimpleSelectOption value="AFTER">After LLM</SimpleSelectOption>
          </SimpleSelect>
        </div>
        <div>
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage defaultMessage="Action" description="Label for guardrail action selector" />
          </Typography.Text>
          <SimpleSelect
            id="guardrail-action-select"
            componentId="mlflow.gateway.guardrails.action-select"
            value={action}
            onChange={({ target }) => {
              if (target.value !== action) setAction(target.value as GuardrailAction);
            }}
          >
            <SimpleSelectOption value="VALIDATION">Block</SimpleSelectOption>
            <SimpleSelectOption value="SANITIZATION">Sanitize</SimpleSelectOption>
          </SimpleSelect>
        </div>
      </div>

      {/* Create custom link */}
      <div css={{ marginTop: theme.spacing.md }}>
        <Button
          componentId="mlflow.gateway.guardrails.create-custom"
          type="link"
          size="small"
          onClick={() => {
            // TODO: navigate to custom guardrail creation (follow-up PR)
          }}
        >
          <FormattedMessage
            defaultMessage="+ Create custom guardrail"
            description="Link to create a custom guardrail"
          />
        </Button>
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

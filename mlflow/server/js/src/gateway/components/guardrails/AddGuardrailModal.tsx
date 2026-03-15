import { useState, useCallback, useMemo, useEffect, useRef, type ReactNode } from 'react';
import {
  Alert,
  Button,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { fetchOrFail, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { useAddGuardrail } from '../../hooks/useAddGuardrail';
import { useUpdateGuardrail } from '../../hooks/useUpdateGuardrail';
import { GatewayApi } from '../../api';
import type { Guardrail, GuardrailHook, GuardrailOperation, GuardrailScorerConfig, TestGuardrailResponse } from '../../types';
import { EndpointSelector } from '@mlflow/mlflow/src/experiment-tracking/components/EndpointSelector';
import {
  formatGatewayModelFromEndpoint,
  getEndpointNameFromGatewayModel,
  ModelProvider,
  getModelProvider,
} from '../../utils/gatewayUtils';

// ─── Guardrails AI icon ─────────────────────────────────────────────────────

function GuardrailsAIIcon({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M12 2L3 7v6c0 5.25 3.83 10.16 9 11.33C17.17 23.16 21 18.25 21 13V7l-9-5z"
        fill="#15B8A6"
      />
      <text
        x="12"
        y="16.5"
        textAnchor="middle"
        fontFamily="Arial, sans-serif"
        fontWeight="bold"
        fontSize="11"
        fill="white"
      >
        G
      </text>
    </svg>
  );
}

// ─── Builtin judges (GuardrailsScorer subclasses from Guardrails AI) ────────

interface BuiltinJudgeInfo {
  name: string;
  label: string;
  description: string;
  variables: ('inputs' | 'outputs')[];
  emoji: string;
}

const BUILTIN_JUDGES: BuiltinJudgeInfo[] = [
  {
    name: 'ToxicLanguage',
    label: 'Toxic Language',
    description: 'Detects toxic, offensive, or harmful content using NLP models',
    variables: ['inputs', 'outputs'],
    emoji: '\u{1F6AB}',
  },
  {
    name: 'NSFWText',
    label: 'NSFW Text',
    description: 'Detects inappropriate or explicit content not suitable for professional settings',
    variables: ['inputs', 'outputs'],
    emoji: '\u{1F6A8}',
  },
  {
    name: 'DetectJailbreak',
    label: 'Detect Jailbreak',
    description: 'Detects jailbreak or prompt injection attempts using a BERT-based classifier',
    variables: ['inputs', 'outputs'],
    emoji: '\u{1F6E1}',
  },
  {
    name: 'DetectPII',
    label: 'Detect PII',
    description: 'Detects personally identifiable information (email, phone, names, etc.)',
    variables: ['inputs', 'outputs'],
    emoji: '\u{1F464}',
  },
  {
    name: 'SecretsPresent',
    label: 'Secrets Present',
    description: 'Detects API keys, tokens, passwords, or other sensitive credentials',
    variables: ['inputs', 'outputs'],
    emoji: '\u{1F511}',
  },
  {
    name: 'GibberishText',
    label: 'Gibberish Text',
    description: 'Detects gibberish, incoherent, or nonsensical text in LLM outputs',
    variables: ['inputs', 'outputs'],
    emoji: '\u{1F4AC}',
  },
];

function getCompatibleBuiltins(hook: GuardrailHook, operation: GuardrailOperation): BuiltinJudgeInfo[] {
  // All GuardrailsScorer subclasses return yes/no → only compatible with VALIDATION
  if (operation === 'MUTATION') return [];
  const requiredVar = hook === 'PRE' ? 'inputs' : 'outputs';
  return BUILTIN_JUDGES.filter((j) => j.variables.includes(requiredVar));
}

// ─── Default prompts for custom judges ──────────────────────────────────────

function getDefaultPrompt(hook: GuardrailHook, operation: GuardrailOperation): string {
  const variable = hook === 'PRE' ? '{{inputs}}' : '{{outputs}}';
  const dataLabel = hook === 'PRE' ? 'user input' : 'model output';
  if (operation === 'VALIDATION') {
    return (
      `Analyze the following ${dataLabel} for personally identifiable information (PII) ` +
      `such as names, email addresses, phone numbers, SSNs, credit card numbers, ` +
      `physical addresses, or other identifying data:\n\n` +
      `${variable}\n\n` +
      `Answer "yes" if the text is safe (no PII detected), or "no" if PII is found.`
    );
  }
  return (
    `Rewrite the following ${dataLabel} to redact any personally identifiable information (PII) ` +
    `such as names, email addresses, phone numbers, SSNs, credit card numbers, ` +
    `or physical addresses. Replace each PII instance with [REDACTED]:\n\n${variable}`
  );
}

// ─── Registered scorer fetching ─────────────────────────────────────────────

interface RegisteredScorer {
  scorer_name: string;
  scorer_version: number;
  experiment_id: string;
  serialized_scorer: string;
}

function useRegisteredScorersQuery() {
  return useQuery<RegisteredScorer[], Error>(['guardrail_registered_scorers'], {
    queryFn: async () => {
      const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/scorers/list?experiment_id=0'));
      const data = await res.json();
      return data.scorers ?? [];
    },
    retry: false,
    staleTime: 30000,
  });
}

/**
 * Check if a registered scorer is compatible with the given hook based on
 * template variables in its serialized definition.
 * - PRE hook needs {{inputs}} (or {{conversation}})
 * - POST hook needs {{outputs}}
 * If we can't parse or find variables, assume compatible.
 */
function isRegisteredScorerCompatible(scorer: RegisteredScorer, hook: GuardrailHook): boolean {
  try {
    const parsed = JSON.parse(scorer.serialized_scorer);
    // make_judge stores instructions in instructions_judge_pydantic_data.instructions
    const instructions: string | undefined =
      parsed?.instructions_judge_pydantic_data?.instructions ??
      parsed?.instructions ??
      parsed?.judge_prompt;
    if (!instructions) return true; // can't determine, show it
    const hasInputs = instructions.includes('{{inputs}}') || instructions.includes('{{ inputs }}');
    const hasOutputs = instructions.includes('{{outputs}}') || instructions.includes('{{ outputs }}');
    const hasConversation =
      instructions.includes('{{conversation}}') || instructions.includes('{{ conversation }}');
    const hasTrace = instructions.includes('{{trace}}') || instructions.includes('{{ trace }}');
    // If it references both or trace/conversation, it's compatible with either hook
    if (hasTrace || hasConversation) return true;
    if (hook === 'PRE') return hasInputs;
    return hasOutputs;
  } catch {
    return true; // can't parse, show it
  }
}

// ─── Form types ─────────────────────────────────────────────────────────────

type ScorerSource = 'builtin' | 'registered' | 'custom' | 'regex' | 'code';

interface FormData {
  scorerName: string;
  hook: GuardrailHook;
  operation: GuardrailOperation;
  scorerSource: ScorerSource;
  builtinJudge: string;
  registeredScorer: string;
  guidelines: string;
  prompt: string;
  model: string;
  regexPattern: string;
}

function makeInitialForm(): FormData {
  return {
    scorerName: '',
    hook: 'PRE',
    operation: 'VALIDATION',
    scorerSource: 'builtin',
    builtinJudge: '',
    registeredScorer: '',
    guidelines: '',
    prompt: getDefaultPrompt('PRE', 'VALIDATION'),
    model: '',
    regexPattern: '',
  };
}

// ─── Guardrail card component ───────────────────────────────────────────────

function GuardrailCard({
  icon,
  label,
  description,
  selected,
  onClick,
  tag,
}: {
  icon: ReactNode;
  label: string;
  description: string;
  selected: boolean;
  onClick: () => void;
  tag?: ReactNode;
}) {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs,
        padding: theme.spacing.md,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${selected ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border}`,
        backgroundColor: selected ? theme.colors.actionPrimaryBackgroundDefault + '08' : theme.colors.backgroundPrimary,
        cursor: 'pointer',
        transition: 'border-color 0.15s, background-color 0.15s, box-shadow 0.15s',
        boxShadow: selected ? `0 0 0 1px ${theme.colors.actionPrimaryBackgroundDefault}` : 'none',
        '&:hover': {
          borderColor: theme.colors.actionPrimaryBackgroundDefault,
          backgroundColor: theme.colors.actionPrimaryBackgroundDefault + '05',
        },
        minHeight: 80,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <span css={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>{icon}</span>
        <Typography.Text bold css={{ flex: 1 }}>
          {label}
        </Typography.Text>
        {tag && <Tag componentId={`mlflow.gateway.guardrail-card.tag.${label}`}>{tag}</Tag>}
      </div>
      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
        {description}
      </Typography.Text>
    </div>
  );
}

// ─── Top-level tab buttons: Catalog | Create New ────────────────────────────

type TopTab = 'catalog' | 'create-new';

function topTabFromSource(source: ScorerSource): TopTab {
  return source === 'custom' || source === 'code' ? 'create-new' : 'catalog';
}

function TopTabs({ value, onChange }: { value: TopTab; onChange: (v: TopTab) => void }) {
  const { theme } = useDesignSystemTheme();

  const tabs: { key: TopTab; label: string }[] = [
    { key: 'catalog', label: 'Catalog' },
    { key: 'create-new', label: 'Create New' },
  ];

  return (
    <div
      css={{
        display: 'flex',
        gap: 0,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
        overflow: 'hidden',
      }}
    >
      {tabs.map((tab) => (
        <button
          key={tab.key}
          type="button"
          onClick={() => onChange(tab.key)}
          css={{
            flex: 1,
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
            border: 'none',
            borderRight: tab.key !== 'create-new' ? `1px solid ${theme.colors.border}` : 'none',
            backgroundColor:
              value === tab.key ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.backgroundPrimary,
            color: value === tab.key ? '#fff' : theme.colors.textPrimary,
            cursor: 'pointer',
            fontWeight: value === tab.key ? 600 : 400,
            fontSize: theme.typography.fontSizeSm,
            transition: 'background-color 0.15s, color 0.15s',
            '&:hover': {
              backgroundColor:
                value === tab.key
                  ? theme.colors.actionPrimaryBackgroundDefault
                  : theme.colors.actionPrimaryBackgroundDefault + '10',
            },
          }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

// ─── Create-New type selector cards ─────────────────────────────────────────

function CreateNewTypeSelector({
  value,
  onChange,
}: {
  value: 'custom' | 'code';
  onChange: (v: 'custom' | 'code') => void;
}) {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: theme.spacing.sm }}>
      <GuardrailCard
        icon={<span css={{ fontSize: 20 }}>{'🤖'}</span>}
        label="LLM Judge"
        description="Write natural-language instructions. An LLM evaluates requests or responses against them."
        selected={value === 'custom'}
        onClick={() => onChange('custom')}
      />
      <GuardrailCard
        icon={<span css={{ fontSize: 20 }}>{'</>'}</span>}
        label="Code-based"
        description="Write arbitrary Python logic — regex, rule engines, external APIs, etc."
        selected={value === 'code'}
        onClick={() => onChange('code')}
      />
    </div>
  );
}

// ─── Model / endpoint selector for custom judges ────────────────────────────

function ModelSelector({ model, onModelChange }: { model: string; onModelChange: (value: string) => void }) {
  const { theme } = useDesignSystemTheme();
  const [providerMode, setProviderMode] = useState<ModelProvider>(
    () => (model ? getModelProvider(model) : ModelProvider.GATEWAY),
  );

  const endpointName = getEndpointNameFromGatewayModel(model);

  if (providerMode === ModelProvider.OTHER) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <Typography.Text bold>Model</Typography.Text>
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          Enter a model identifier (e.g., openai:/gpt-4.1-mini). Leave empty to use the default model.
        </Typography.Text>
        <Input
          componentId="mlflow.gateway.guardrail-modal.model-input"
          value={model}
          onChange={(e) => onModelChange(e.target.value)}
          placeholder="openai:/gpt-4.1-mini"
        />
        <Typography.Link
          componentId="mlflow.gateway.guardrail-modal.switch-to-endpoint"
          onClick={() => {
            setProviderMode(ModelProvider.GATEWAY);
            onModelChange('');
          }}
          css={{ cursor: 'pointer', fontSize: theme.typography.fontSizeSm }}
        >
          ← Use an endpoint instead
        </Typography.Link>
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Typography.Text bold>Model</Typography.Text>
      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
        Select an endpoint to use for this guardrail judge. Leave empty to use the default model.
      </Typography.Text>
      <EndpointSelector
        currentEndpointName={endpointName}
        onEndpointSelect={(name) => onModelChange(formatGatewayModelFromEndpoint(name))}
        componentIdPrefix="mlflow.gateway.guardrail-modal.endpoint"
        placeholder="Default model"
      />
      <Typography.Text color="secondary" size="sm">
        Or{' '}
        <Typography.Link
          componentId="mlflow.gateway.guardrail-modal.switch-to-manual"
          onClick={() => {
            setProviderMode(ModelProvider.OTHER);
            onModelChange('');
          }}
          css={{ cursor: 'pointer' }}
        >
          enter a model identifier
        </Typography.Link>
      </Typography.Text>
    </div>
  );
}

// ─── Modal component ────────────────────────────────────────────────────────

interface TraceEntry {
  trace_id: string;
  request_preview?: string;
  response_preview?: string;
  request_time?: string;
  state?: string;
}

interface GuardrailModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  endpointName?: string;
  editingGuardrail?: Guardrail | null;
  experimentId?: string;
}

export const GuardrailModal = ({ open, onClose, onSuccess, endpointName, editingGuardrail, experimentId }: GuardrailModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isEditMode = !!editingGuardrail;

  const [formData, setFormData] = useState<FormData>(makeInitialForm);
  const [searchQuery, setSearchQuery] = useState('');
  const { data: registeredScorers, isLoading: isLoadingScorers } = useRegisteredScorersQuery();

  const {
    mutateAsync: addGuardrail,
    isLoading: isAdding,
    error: addError,
    reset: resetAdd,
  } = useAddGuardrail();
  const {
    mutateAsync: updateGuardrail,
    isLoading: isUpdating,
    error: updateError,
    reset: resetUpdate,
  } = useUpdateGuardrail();

  const isLoading = isAdding || isUpdating;
  const mutationError = addError || updateError;

  // ─── Test state ──────────────────────────────────────────────────────────
  const [showTest, setShowTest] = useState(false);
  const [testInputMode, setTestInputMode] = useState<'trace' | 'manual'>('manual');
  const [testManualText, setTestManualText] = useState('');
  const [traces, setTraces] = useState<TraceEntry[]>([]);
  const [tracesLoading, setTracesLoading] = useState(false);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TestGuardrailResponse | null>(null);
  const [testing, setTesting] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch traces when test panel is shown
  useEffect(() => {
    if (!showTest || !experimentId) return;
    const fetchTraces = async () => {
      setTracesLoading(true);
      try {
        const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/traces/search'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: experimentId } }],
            max_results: 20,
            order_by: ['timestamp_ms DESC'],
          }),
        });
        const data = (await res.json()) as { traces?: TraceEntry[] };
        setTraces(data.traces ?? []);
      } catch {
        // silently fail
      } finally {
        setTracesLoading(false);
      }
    };
    fetchTraces();
  }, [showTest, experimentId]);

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Build the test request body — either from existing guardrail or from form data
  const buildTestRequest = useCallback((): Partial<import('../../types').TestGuardrailRequest> => {
    if (isEditMode && editingGuardrail) {
      return { guardrail_id: editingGuardrail.guardrail_id };
    }
    // Inline config from form data
    const config: GuardrailScorerConfig = {};
    if (formData.scorerSource === 'builtin') {
      config.builtin_scorer = formData.builtinJudge;
    } else if (formData.scorerSource === 'registered') {
      config.registered_scorer = formData.registeredScorer;
      const scorer = registeredScorers?.find((s) => s.scorer_name === formData.registeredScorer);
      if (scorer) {
        config.experiment_id = scorer.experiment_id;
        config.scorer_version = scorer.scorer_version;
      }
    } else if (formData.scorerSource === 'regex') {
      config.builtin_scorer = 'RegexMatch';
      config.regex_pattern = formData.regexPattern;
    } else {
      config.prompt = formData.prompt;
      config.response_schema =
        formData.operation === 'VALIDATION'
          ? 'yes_no'
          : formData.hook === 'PRE'
            ? 'chat_request'
            : 'chat_response';
      if (formData.model) {
        config.model = formData.model;
      }
    }
    return {
      scorer_name: formData.scorerName,
      hook: formData.hook,
      operation: formData.operation,
      config,
    };
  }, [isEditMode, editingGuardrail, formData, registeredScorers]);

  const handleTest = useCallback(async () => {
    setTesting(true);
    setTestResult(null);
    setTestError(null);
    if (pollRef.current) clearInterval(pollRef.current);

    try {
      const base = buildTestRequest();
      if (testInputMode === 'trace' && selectedTraceId && experimentId) {
        const result = await GatewayApi.testGuardrail({
          ...base,
          trace_id: selectedTraceId,
          experiment_id: experimentId,
        });
        setTestResult(result);
      } else if (testInputMode === 'manual' && testManualText) {
        const result = await GatewayApi.testGuardrail({
          ...base,
          text: testManualText,
        });
        setTestResult(result);
      } else {
        setTestError('Please provide text or select a trace to test against.');
      }
    } catch (e: any) {
      setTestError(e.message || 'Test failed');
    } finally {
      setTesting(false);
    }
  }, [buildTestRequest, testInputMode, selectedTraceId, experimentId, testManualText]);

  // Populate form when editing
  useEffect(() => {
    if (!open) return;
    if (editingGuardrail) {
      const config = editingGuardrail.config;
      const isBuiltin = BUILTIN_JUDGES.some((j) => j.name === config?.builtin_scorer);
      const isRegex = config?.builtin_scorer === 'RegexMatch';
      const isRegistered = !!config?.registered_scorer;
      setFormData({
        scorerName: editingGuardrail.scorer_name,
        hook: editingGuardrail.hook,
        operation: editingGuardrail.operation,
        scorerSource: isRegex ? 'regex' : isBuiltin ? 'builtin' : isRegistered ? 'registered' : 'custom',
        builtinJudge: isRegex ? '' : (config?.builtin_scorer ?? ''),
        registeredScorer: config?.registered_scorer ?? '',
        guidelines: config?.guidelines ?? '',
        prompt: config?.prompt ?? getDefaultPrompt(editingGuardrail.hook, editingGuardrail.operation),
        model: config?.model ?? '',
        regexPattern: config?.regex_pattern ?? '',
      });
    } else {
      setFormData(makeInitialForm());
    }
    setSearchQuery('');
    setShowTest(false);
    setTestResult(null);
    setTestError(null);
    setSelectedTraceId(null);
    setTestManualText('');
    setTestInputMode('manual');
  }, [open, editingGuardrail, experimentId]);

  const resetMutation = useCallback(() => {
    resetAdd();
    resetUpdate();
  }, [resetAdd, resetUpdate]);

  const handleClose = useCallback(() => {
    setFormData(makeInitialForm());
    setSearchQuery('');
    resetMutation();
    onClose();
  }, [onClose, resetMutation]);

  const handleFieldChange = useCallback(
    <K extends keyof FormData>(field: K, value: FormData[K]) => {
      setFormData((prev) => {
        const next = { ...prev, [field]: value };
        if (field === 'builtinJudge' && prev.scorerSource === 'builtin') {
          next.scorerName = String(value);
        }
        if (field === 'registeredScorer' && prev.scorerSource === 'registered') {
          next.scorerName = String(value);
        }
        if (field === 'regexPattern') {
          // Auto-generate a scorer name from the pattern (slug-ify it)
          const slug = String(value)
            .replace(/[^a-zA-Z0-9]+/g, '-')
            .replace(/^-+|-+$/g, '')
            .toLowerCase()
            .slice(0, 40) || 'regex-guardrail';
          next.scorerName = slug;
        }
        if (field === 'scorerSource') {
          if (value === 'builtin') {
            next.scorerName = next.builtinJudge;
          } else if (value === 'registered' && next.registeredScorer) {
            next.scorerName = next.registeredScorer;
          } else if (value === 'custom') {
            next.scorerName = prev.scorerName || '';
            next.prompt = getDefaultPrompt(next.hook, next.operation);
          } else if (value === 'regex') {
            next.scorerName = prev.scorerName || '';
            next.operation = 'VALIDATION'; // regex only supports block
          } else if (value === 'code') {
            next.scorerName = '';
          }
        }
        // When hook or operation changes, clear selections that may no longer be compatible
        if (field === 'hook' || field === 'operation') {
          if (prev.scorerSource === 'builtin') {
            // Builtins only support VALIDATION; clear selection if switching to MUTATION
            const newOp = (field === 'operation' ? value : next.operation) as GuardrailOperation;
            if (newOp === 'MUTATION') {
              next.builtinJudge = '';
              next.scorerName = '';
            }
          }
          if (prev.scorerSource === 'custom') {
            const oldDefault = getDefaultPrompt(prev.hook, prev.operation);
            if (prev.prompt === oldDefault) {
              next.prompt = getDefaultPrompt(next.hook, next.operation);
            }
          }
        }
        return next;
      });
      resetMutation();
    },
    [resetMutation],
  );

  const compatibleBuiltins = useMemo(
    () => getCompatibleBuiltins(formData.hook, formData.operation),
    [formData.hook, formData.operation],
  );

  const filteredBuiltins = useMemo(() => {
    if (!searchQuery.trim()) return compatibleBuiltins;
    const q = searchQuery.toLowerCase();
    return compatibleBuiltins.filter(
      (j) => j.label.toLowerCase().includes(q) || j.description.toLowerCase().includes(q),
    );
  }, [compatibleBuiltins, searchQuery]);

  const filteredRegistered = useMemo(() => {
    if (!registeredScorers) return [];
    // Filter by hook compatibility (template variables)
    const compatible = registeredScorers.filter((s) => isRegisteredScorerCompatible(s, formData.hook));
    if (!searchQuery.trim()) return compatible;
    const q = searchQuery.toLowerCase();
    return compatible.filter((s) => s.scorer_name.toLowerCase().includes(q));
  }, [registeredScorers, formData.hook, searchQuery]);

  const isFormValid = useMemo(() => {
    if (formData.scorerSource === 'builtin' && !formData.builtinJudge) return false;
    if (formData.scorerSource === 'custom' && !formData.scorerName.trim()) return false;
    if (formData.scorerSource === 'custom' && !formData.prompt.trim()) return false;
    if (formData.scorerSource === 'registered' && !formData.registeredScorer) return false;
    if (formData.scorerSource === 'regex' && !formData.scorerName.trim()) return false;
    if (formData.scorerSource === 'regex' && !formData.regexPattern.trim()) return false;
    if (formData.scorerSource === 'code') return false; // code tab is guide-only, no submission
    return true;
  }, [formData]);

  const handleSubmit = useCallback(async () => {
    if (!isFormValid) return;

    const config: GuardrailScorerConfig = {};

    if (formData.scorerSource === 'builtin') {
      config.builtin_scorer = formData.builtinJudge;
    } else if (formData.scorerSource === 'registered') {
      config.registered_scorer = formData.registeredScorer;
      const scorer = registeredScorers?.find((s) => s.scorer_name === formData.registeredScorer);
      if (scorer) {
        config.experiment_id = scorer.experiment_id;
        config.scorer_version = scorer.scorer_version;
      }
    } else if (formData.scorerSource === 'regex') {
      config.builtin_scorer = 'RegexMatch';
      config.regex_pattern = formData.regexPattern;
    } else {
      config.prompt = formData.prompt;
      config.response_schema =
        formData.operation === 'VALIDATION'
          ? 'yes_no'
          : formData.hook === 'PRE'
            ? 'chat_request'
            : 'chat_response';
      if (formData.model) {
        config.model = formData.model;
      }
    }

    if (isEditMode && editingGuardrail) {
      await updateGuardrail({
        guardrail_id: editingGuardrail.guardrail_id,
        scorer_name: formData.scorerName,
        hook: formData.hook,
        operation: formData.operation,
        config,
      }).then(() => {
        handleClose();
        onSuccess?.();
      });
    } else {
      await addGuardrail({
        scorer_name: formData.scorerName,
        hook: formData.hook,
        operation: formData.operation,
        endpoint_name: endpointName,
        config,
      }).then(() => {
        handleClose();
        onSuccess?.();
      });
    }
  }, [
    isFormValid,
    formData,
    isEditMode,
    editingGuardrail,
    addGuardrail,
    updateGuardrail,
    endpointName,
    handleClose,
    onSuccess,
    registeredScorers,
  ]);

  const errorMessage = useMemo((): string | null => {
    if (!mutationError) return null;
    const message = (mutationError as Error).message;
    return message.length > 200
      ? intl.formatMessage({
          defaultMessage: 'An error occurred. Please try again.',
          description: 'Generic error message for guardrail modal',
        })
      : message;
  }, [mutationError, intl]);

  const promptHint = formData.hook === 'PRE' ? '{{inputs}}' : '{{outputs}}';

  return (
    <Modal
      componentId="mlflow.gateway.guardrail-modal"
      title={
        isEditMode
          ? intl.formatMessage({ defaultMessage: 'Edit Guardrail', description: 'Title for edit guardrail modal' })
          : intl.formatMessage({ defaultMessage: 'Add Guardrail', description: 'Title for add guardrail modal' })
      }
      visible={open}
      onCancel={handleClose}
      onOk={handleSubmit}
      okText={
        isEditMode
          ? intl.formatMessage({ defaultMessage: 'Save', description: 'Save guardrail button text' })
          : intl.formatMessage({ defaultMessage: 'Add', description: 'Add guardrail button text' })
      }
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Cancel button text',
      })}
      confirmLoading={isLoading}
      okButtonProps={{ disabled: !isFormValid }}
      size="wide"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        {errorMessage && (
          <Alert
            componentId="mlflow.gateway.guardrail-modal.error"
            type="error"
            message={errorMessage}
            closable={false}
          />
        )}

        {/* Stage and Action side by side */}
        <div css={{ display: 'flex', gap: theme.spacing.md }}>
          <div css={{ flex: 1, display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Stage" description="Guardrail stage label" />
            </Typography.Text>
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              <FormattedMessage
                defaultMessage="When the guardrail runs relative to LLM invocation"
                description="Stage description"
              />
            </Typography.Text>
            <SimpleSelect
              id="guardrail-hook"
              componentId="mlflow.gateway.guardrail-modal.hook"
              value={formData.hook}
              onChange={({ target }) => handleFieldChange('hook', target.value as GuardrailHook)}
            >
              <SimpleSelectOption value="PRE">Pre-invocation</SimpleSelectOption>
              <SimpleSelectOption value="POST">Post-invocation</SimpleSelectOption>
            </SimpleSelect>
          </div>

          <div css={{ flex: 1, display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Action" description="Guardrail action label" />
            </Typography.Text>
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              <FormattedMessage
                defaultMessage="Whether the guardrail blocks or modifies requests"
                description="Action description"
              />
            </Typography.Text>
            <SimpleSelect
              id="guardrail-operation"
              componentId="mlflow.gateway.guardrail-modal.operation"
              value={formData.operation}
              onChange={({ target }) => handleFieldChange('operation', target.value as GuardrailOperation)}
            >
              <SimpleSelectOption value="VALIDATION">Block</SimpleSelectOption>
              <SimpleSelectOption value="MUTATION">Modify</SimpleSelectOption>
            </SimpleSelect>
          </div>
        </div>

        {/* ─── Top-level tabs: Catalog | Create New ─── */}
        <TopTabs
          value={topTabFromSource(formData.scorerSource)}
          onChange={(tab) => {
            if (tab === 'catalog') handleFieldChange('scorerSource', 'builtin');
            else handleFieldChange('scorerSource', 'custom');
          }}
        />

        {/* ─── CATALOG: Guardrails AI + Regex + Registered ─── */}
        {topTabFromSource(formData.scorerSource) === 'catalog' && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            {/* Search */}
            <Input
              componentId="mlflow.gateway.guardrail-modal.search"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={intl.formatMessage({
                defaultMessage: 'Search guardrails...',
                description: 'Search guardrails placeholder',
              })}
              allowClear
            />

            {/* Builtin section */}
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <Typography.Text bold css={{ fontSize: theme.typography.fontSizeSm }}>
                  Builtin
                </Typography.Text>
              </div>
              {formData.operation === 'MUTATION' ? (
                <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                  <FormattedMessage
                    defaultMessage="Catalog guardrails only support Block action. Switch to Block or use Create New for Modify."
                    description="No catalog guardrails for modify"
                  />
                </Typography.Text>
              ) : (
                <div css={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: theme.spacing.sm }}>
                  {/* Builtin judge cards */}
                  {filteredBuiltins.map((j) => (
                    <GuardrailCard
                      key={j.name}
                      icon={<span css={{ fontSize: 20, lineHeight: 1 }}>{j.emoji}</span>}
                      label={j.label}
                      description={j.description}
                      selected={formData.scorerSource === 'builtin' && formData.builtinJudge === j.name}
                      tag={<span css={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><GuardrailsAIIcon size={14} /> BUILTIN</span>}
                      onClick={() => {
                        handleFieldChange('scorerSource', 'builtin');
                        handleFieldChange('builtinJudge', j.name);
                      }}
                    />
                  ))}
                  {/* Regex card — part of Guardrails AI catalog */}
                  {(!searchQuery.trim() || 'regex match'.includes(searchQuery.toLowerCase())) && (
                    <GuardrailCard
                      icon={<span css={{ fontSize: 20, lineHeight: 1 }}>{'🔍'}</span>}
                      label="Regex Match"
                      description="Block requests or responses matching a regular expression pattern (e.g. SSNs, emails, credit cards)"
                      selected={formData.scorerSource === 'regex'}
                      tag={<span css={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><GuardrailsAIIcon size={14} /> BUILTIN</span>}
                      onClick={() => handleFieldChange('scorerSource', 'regex')}
                    />
                  )}
                  {filteredBuiltins.length === 0 && searchQuery.trim() && (
                    <div css={{ gridColumn: '1 / -1', padding: theme.spacing.sm }}>
                      <Typography.Text color="secondary">
                        <FormattedMessage defaultMessage="No matching guardrails found." description="No matching builtin guardrails" />
                      </Typography.Text>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Regex config — shown inline when regex card is selected */}
            {formData.scorerSource === 'regex' && (
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.sm,
                  padding: theme.spacing.md,
                  border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  backgroundColor: `${theme.colors.actionPrimaryBackgroundDefault}06`,
                }}
              >
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold css={{ fontSize: theme.typography.fontSizeSm }}>
                    <FormattedMessage defaultMessage="Pattern" description="Regex pattern label" />
                  </Typography.Text>
                  <Input
                    componentId="mlflow.gateway.guardrail-modal.regex-pattern"
                    value={formData.regexPattern}
                    onChange={(e) => handleFieldChange('regexPattern', e.target.value)}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'e.g., \\d{3}-\\d{2}-\\d{4}  (SSN)',
                      description: 'Regex pattern placeholder',
                    })}
                    css={{ fontFamily: 'monospace' }}
                  />
                </div>
                <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                  Common patterns: SSN <code>\d{'{3}'}-\d{'{2}'}-\d{'{4}'}</code> · email <code>[\w.-]+@[\w.-]+\.\w+</code> · credit card <code>\d{'{4}'}[\s-]?\d{'{4}'}[\s-]?\d{'{4}'}[\s-]?\d{'{4}'}</code>
                </Typography.Text>
              </div>
            )}

            {/* Registered section */}
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, borderTop: `1px solid ${theme.colors.border}`, paddingTop: theme.spacing.sm }}>
                <span css={{ fontSize: 14 }}>⚙️</span>
                <Typography.Text bold css={{ fontSize: theme.typography.fontSizeSm }}>
                  Registered
                </Typography.Text>
              </div>
              {isLoadingScorers ? (
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                  <Spinner size="small" />
                  <FormattedMessage defaultMessage="Loading registered guardrails..." description="Loading registered guardrails" />
                </div>
              ) : filteredRegistered.length > 0 ? (
                <div css={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: theme.spacing.sm }}>
                  {filteredRegistered.map((s) => (
                    <GuardrailCard
                      key={s.scorer_name}
                      icon={<span css={{ fontSize: 20, lineHeight: 1 }}>{'⚙️'}</span>}
                      label={s.scorer_name}
                      description={`v${s.scorer_version} · experiment ${s.experiment_id}`}
                      selected={formData.scorerSource === 'registered' && formData.registeredScorer === s.scorer_name}
                      tag="REGISTERED"
                      onClick={() => {
                        handleFieldChange('scorerSource', 'registered');
                        handleFieldChange('registeredScorer', s.scorer_name);
                      }}
                    />
                  ))}
                </div>
              ) : registeredScorers && registeredScorers.length > 0 ? (
                <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                  {formData.hook === 'PRE'
                    ? "No registered guardrails compatible with pre-invocation. They need '{{inputs}}' in their instructions."
                    : "No registered guardrails compatible with post-invocation. They need '{{outputs}}' in their instructions."}
                </Typography.Text>
              ) : (
                <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                  No registered guardrails yet. Use <strong>Create New → Code-based</strong> to write and register one.
                </Typography.Text>
              )}
            </div>
          </div>
        )}

        {/* ─── CREATE NEW: LLM Judge or Code-based ─── */}
        {topTabFromSource(formData.scorerSource) === 'create-new' && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <CreateNewTypeSelector
              value={formData.scorerSource === 'code' ? 'code' : 'custom'}
              onChange={(v) => handleFieldChange('scorerSource', v)}
            />

            {/* ─── LLM Judge form ─── */}
            {formData.scorerSource === 'custom' && (
              <>
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold>
                    <FormattedMessage defaultMessage="Guardrail name" description="LLM judge guardrail name label" />
                  </Typography.Text>
                  <Input
                    componentId="mlflow.gateway.guardrail-modal.custom-name"
                    value={formData.scorerName}
                    onChange={(e) => handleFieldChange('scorerName', e.target.value)}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'e.g., pii-filter, toxicity-check',
                      description: 'Guardrail name placeholder',
                    })}
                  />
                </div>
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold>
                    <FormattedMessage defaultMessage="Instructions" description="LLM judge instructions label" />
                  </Typography.Text>
                  <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                    <FormattedMessage
                      defaultMessage="Instructions for the LLM guardrail. Use {variable} to reference the {hookType} data."
                      description="LLM judge instructions description"
                      values={{
                        variable: <code>{promptHint}</code>,
                        hookType: formData.hook === 'PRE' ? 'input' : 'output',
                      }}
                    />
                  </Typography.Text>
                  <textarea
                    value={formData.prompt}
                    onChange={(e) => handleFieldChange('prompt', e.target.value)}
                    rows={6}
                    css={{
                      width: '100%',
                      padding: theme.spacing.sm,
                      borderRadius: theme.borders.borderRadiusMd,
                      border: `1px solid ${theme.colors.border}`,
                      fontFamily: 'monospace',
                      fontSize: theme.typography.fontSizeSm,
                      resize: 'vertical',
                      backgroundColor: theme.colors.backgroundPrimary,
                      color: theme.colors.textPrimary,
                    }}
                  />
                  {formData.operation === 'VALIDATION' && (
                    <Alert
                      componentId="mlflow.gateway.guardrail-modal.validation-hint"
                      type="info"
                      message={intl.formatMessage({
                        defaultMessage: 'Block guardrails expect a yes/no response. "no" means the content is rejected.',
                        description: 'Block hint for LLM judge guardrails',
                      })}
                    />
                  )}
                  {formData.operation === 'MUTATION' && (
                    <Alert
                      componentId="mlflow.gateway.guardrail-modal.mutation-hint"
                      type="info"
                      message={intl.formatMessage({
                        defaultMessage: 'Modify guardrails should return the modified content. For pre-invocation, output a ChatRequest. For post-invocation, output a ChatResponse.',
                        description: 'Modify hint for LLM judge guardrails',
                      })}
                    />
                  )}
                </div>
                <ModelSelector
                  model={formData.model}
                  onModelChange={(value) => handleFieldChange('model', value)}
                />
              </>
            )}

            {/* ─── Code-based guide ─── */}
            {formData.scorerSource === 'code' && (
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                <Alert
                  componentId="mlflow.gateway.guardrail-modal.code-info"
                  type="info"
                  closable={false}
                  message="Write a Python class, register it with mlflow.genai.scorers.register_scorer(), then find it in Catalog → Registered."
                />
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold>
                    <FormattedMessage defaultMessage="Example: block requests containing SSNs" description="Code guardrail example header" />
                  </Typography.Text>
              <pre
                css={{
                  backgroundColor: theme.colors.backgroundSecondary,
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  padding: theme.spacing.md,
                  fontSize: theme.typography.fontSizeSm,
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                  margin: 0,
                  color: theme.colors.textPrimary,
                }}
              >{`import re
from mlflow.genai.scorers import scorer

@scorer
def ssn_detector(inputs) -> bool:
    """Returns False (block) when an SSN pattern is found."""
    return not bool(re.search(r"\\d{3}-\\d{2}-\\d{4}", str(inputs)))

# Register it once — then find it in the Registered tab
ssn_detector.register()`}</pre>
            </div>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Text bold>
                <FormattedMessage defaultMessage="Example: redact PII before sending to LLM (Modify)" description="Code guardrail modify example header" />
              </Typography.Text>
              <pre
                css={{
                  backgroundColor: theme.colors.backgroundSecondary,
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  padding: theme.spacing.md,
                  fontSize: theme.typography.fontSizeSm,
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                  margin: 0,
                  color: theme.colors.textPrimary,
                }}
              >{`import re
import copy
from mlflow.genai.scorers import scorer

@scorer
def pii_redactor(inputs):
    """Redacts SSNs from the request before sending to the LLM."""
    modified = copy.deepcopy(inputs)
    for msg in modified.get("messages", []):
        msg["content"] = re.sub(
            r"\\d{3}-\\d{2}-\\d{4}", "[REDACTED]",
            str(msg.get("content", ""))
        )
    return modified  # return modified ChatRequest

# Register it once — then find it in the Registered tab
pii_redactor.register()`}</pre>
            </div>
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              <FormattedMessage
                defaultMessage="After registering, refresh the Registered tab to use your guardrail. See the MLflow docs for more scorer examples."
                description="Code guardrail footer note"
              />
            </Typography.Text>
          </div>
        )}
          </div>
        )}

        {/* ─── Test section ─── */}
        {isFormValid && (
          <div
            css={{
              borderTop: `1px solid ${theme.colors.border}`,
              paddingTop: theme.spacing.md,
            }}
          >
            <div
              css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}
              onClick={() => setShowTest((v) => !v)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setShowTest((v) => !v);
                }
              }}
            >
              <Typography.Text bold>
                <FormattedMessage defaultMessage="Test guardrail" description="Test guardrail section header" />
              </Typography.Text>
              <span css={{ fontSize: 12, color: theme.colors.textSecondary }}>
                {showTest ? '\u25B2' : '\u25BC'}
              </span>
            </div>

            {showTest && (
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, marginTop: theme.spacing.md }}>
                <Alert
                  componentId="mlflow.gateway.guardrail-modal.test-info"
                  type="info"
                  closable={false}
                  message={intl.formatMessage(
                    {
                      defaultMessage:
                        'This is a {hook}-invocation {operation} guardrail. It will test against {dataSource}.',
                      description: 'Info about guardrail test behavior',
                    },
                    {
                      hook: formData.hook === 'PRE' ? 'pre' : 'post',
                      operation: formData.operation.toLowerCase(),
                      dataSource: formData.hook === 'PRE' ? 'request input' : 'response output',
                    },
                  )}
                />

                {/* Input mode tabs */}
                {experimentId && (
                  <div css={{ display: 'flex', gap: theme.spacing.xs }}>
                    <Button
                      componentId="mlflow.gateway.guardrail-modal.test-mode-manual"
                      type={testInputMode === 'manual' ? 'primary' : undefined}
                      size="small"
                      onClick={() => setTestInputMode('manual')}
                    >
                      <FormattedMessage defaultMessage="Manual input" description="Tab for manual text input" />
                    </Button>
                    <Button
                      componentId="mlflow.gateway.guardrail-modal.test-mode-trace"
                      type={testInputMode === 'trace' ? 'primary' : undefined}
                      size="small"
                      onClick={() => setTestInputMode('trace')}
                    >
                      <FormattedMessage defaultMessage="Pick from traces" description="Tab to pick a trace" />
                    </Button>
                  </div>
                )}

                {/* Trace picker */}
                {testInputMode === 'trace' && experimentId && (
                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                    {tracesLoading && (
                      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                        <Spinner size="small" />
                        <FormattedMessage defaultMessage="Loading traces..." description="Loading traces" />
                      </div>
                    )}
                    {!tracesLoading && traces.length === 0 && (
                      <Typography.Text color="secondary">
                        <FormattedMessage
                          defaultMessage="No traces found. Send some requests first, or use manual input."
                          description="No traces found message"
                        />
                      </Typography.Text>
                    )}
                    {!tracesLoading && traces.length > 0 && (
                      <div
                        css={{
                          maxHeight: 200,
                          overflowY: 'auto',
                          border: `1px solid ${theme.colors.border}`,
                          borderRadius: theme.borders.borderRadiusMd,
                        }}
                      >
                        {traces.map((trace) => {
                          const isSelected = selectedTraceId === trace.trace_id;
                          const preview =
                            formData.hook === 'PRE' ? trace.request_preview : trace.response_preview;
                          return (
                            <div
                              key={trace.trace_id}
                              css={{
                                padding: theme.spacing.sm,
                                borderBottom: `1px solid ${theme.colors.border}`,
                                cursor: 'pointer',
                                backgroundColor: isSelected
                                  ? `${theme.colors.actionPrimaryBackgroundDefault}12`
                                  : 'transparent',
                                border: isSelected
                                  ? `2px solid ${theme.colors.actionPrimaryBackgroundDefault}`
                                  : '2px solid transparent',
                                borderRadius: theme.borders.borderRadiusMd,
                                '&:hover': { backgroundColor: theme.colors.tableRowHover },
                                '&:last-child': { borderBottom: 'none' },
                              }}
                              onClick={() => setSelectedTraceId(trace.trace_id)}
                              role="button"
                              tabIndex={0}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' || e.key === ' ') {
                                  e.preventDefault();
                                  setSelectedTraceId(trace.trace_id);
                                }
                              }}
                            >
                              <div
                                css={{
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  alignItems: 'center',
                                  marginBottom: 2,
                                }}
                              >
                                <Typography.Text
                                  css={{ fontSize: theme.typography.fontSizeSm, fontFamily: 'monospace' }}
                                >
                                  {trace.trace_id.substring(0, 12)}...
                                </Typography.Text>
                                {trace.state && (
                                  <span
                                    css={{
                                      fontSize: 11,
                                      padding: '1px 6px',
                                      borderRadius: 4,
                                      backgroundColor: trace.state === 'OK' ? '#52c41a20' : '#ff4d4f20',
                                      color: trace.state === 'OK' ? '#52c41a' : '#ff4d4f',
                                      fontWeight: 600,
                                    }}
                                  >
                                    {trace.state}
                                  </span>
                                )}
                              </div>
                              <Typography.Text
                                color="secondary"
                                css={{
                                  fontSize: theme.typography.fontSizeSm,
                                  display: '-webkit-box',
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: 'vertical',
                                  overflow: 'hidden',
                                  wordBreak: 'break-word',
                                }}
                              >
                                {preview
                                  ? preview.length > 150
                                    ? `${preview.substring(0, 150)}...`
                                    : preview
                                  : '(no preview)'}
                              </Typography.Text>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}

                {/* Manual input */}
                {testInputMode === 'manual' && (
                  <Input.TextArea
                    componentId="mlflow.guardrails.add-modal.test-manual-input"
                    value={testManualText}
                    onChange={(e) => setTestManualText(e.target.value)}
                    rows={3}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Enter the text you want to test the guardrail against...',
                      description: 'Placeholder for manual test input',
                    })}
                  />
                )}

                {/* Run test button */}
                <Button
                  componentId="mlflow.gateway.guardrail-modal.run-test"
                  type="primary"
                  onClick={handleTest}
                  loading={testing}
                  disabled={testInputMode === 'trace' ? !selectedTraceId : !testManualText}
                >
                  <FormattedMessage defaultMessage="Run test" description="Run test button" />
                </Button>

                {/* Test error */}
                {testError && (
                  <Alert
                    componentId="mlflow.gateway.guardrail-modal.test-error"
                    type="error"
                    closable={false}
                    message={testError}
                  />
                )}

                {/* Test result */}
                {testResult && (
                  <TestResultDisplay result={testResult} />
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </Modal>
  );
};

// ─── Helpers for readable text display ───────────────────────────────────────

/**
 * Try to parse JSON and extract human-readable content.
 * Handles chat messages format: [{"role":"user","content":"..."}]
 * and single-string JSON values.
 */
function formatDisplayText(raw: string): { formatted: string; isStructured: boolean } {
  try {
    const parsed = JSON.parse(raw);

    // Array of chat messages: [{ role, content }, ...]
    if (Array.isArray(parsed)) {
      const lines = parsed
        .filter((msg: any) => msg?.role && msg?.content)
        .map((msg: any) => {
          const content =
            typeof msg.content === 'string'
              ? msg.content
              : Array.isArray(msg.content)
                ? msg.content
                    .filter((p: any) => p?.type === 'text')
                    .map((p: any) => p.text)
                    .join('\n')
                : JSON.stringify(msg.content);
          return `[${msg.role}] ${content}`;
        });
      if (lines.length > 0) {
        return { formatted: lines.join('\n\n'), isStructured: true };
      }
    }

    // Single object with "content" (e.g. a single message)
    if (parsed && typeof parsed === 'object' && 'content' in parsed) {
      const content = typeof parsed.content === 'string' ? parsed.content : JSON.stringify(parsed.content, null, 2);
      return { formatted: parsed.role ? `[${parsed.role}] ${content}` : content, isStructured: true };
    }

    // Other JSON — pretty-print it
    if (typeof parsed === 'string') {
      return { formatted: parsed, isStructured: false };
    }
    return { formatted: JSON.stringify(parsed, null, 2), isStructured: true };
  } catch {
    // Not JSON — return as-is
    return { formatted: raw, isStructured: false };
  }
}

// ─── Test result display ─────────────────────────────────────────────────────

const TestResultDisplay = ({ result }: { result: TestGuardrailResponse }) => {
  const { theme } = useDesignSystemTheme();
  const isPassed = result.result.score === 'yes';
  const isMutation = result.guardrail.operation === 'MUTATION';
  const displayInput = result.input_text ? formatDisplayText(result.input_text) : null;
  const displayModified = result.result.modified_text ? formatDisplayText(result.result.modified_text) : null;

  return (
    <div
      css={{
        border: `2px solid ${isPassed ? '#52c41a' : '#ff4d4f'}`,
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          backgroundColor: isPassed ? '#52c41a12' : '#ff4d4f12',
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          borderBottom: `1px solid ${isPassed ? '#52c41a40' : '#ff4d4f40'}`,
        }}
      >
        <span css={{ fontSize: 20 }}>{isPassed ? '\u2705' : '\u274C'}</span>
        <Typography.Title level={4} css={{ margin: '0 !important' }}>
          {isPassed ? (
            <FormattedMessage defaultMessage="Passed" description="Test passed label" />
          ) : (
            <FormattedMessage defaultMessage="Failed" description="Test failed label" />
          )}
        </Typography.Title>
        <span
          css={{
            marginLeft: 'auto',
            fontSize: theme.typography.fontSizeSm,
            padding: '2px 8px',
            borderRadius: 4,
            backgroundColor: isPassed ? '#52c41a20' : '#ff4d4f20',
            color: isPassed ? '#52c41a' : '#ff4d4f',
            fontWeight: 600,
          }}
        >
          score: {result.result.score}
        </span>
      </div>

      <div css={{ padding: theme.spacing.md }}>
        {result.result.rationale && (
          <>
            <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Rationale" description="Rationale label in test result" />
            </Typography.Text>
            <Typography.Text css={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {result.result.rationale}
            </Typography.Text>
          </>
        )}

        {displayInput && (
          <>
            <Typography.Text
              bold
              css={{ display: 'block', marginTop: theme.spacing.md, marginBottom: theme.spacing.xs }}
            >
              <FormattedMessage defaultMessage="Tested against" description="Input text label in test result" />
            </Typography.Text>
            <div
              css={{
                padding: theme.spacing.sm,
                backgroundColor: theme.colors.backgroundSecondary,
                borderRadius: theme.borders.borderRadiusMd,
                border: `1px solid ${theme.colors.border}`,
                maxHeight: 120,
                overflowY: 'auto',
                fontSize: theme.typography.fontSizeSm,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                fontFamily: displayInput.isStructured ? 'inherit' : 'monospace',
                lineHeight: 1.5,
              }}
            >
              {displayInput.formatted.length > 800
                ? `${displayInput.formatted.substring(0, 800)}...`
                : displayInput.formatted}
            </div>
          </>
        )}

        {isMutation && displayModified && (
          <>
            <Typography.Text
              bold
              css={{ display: 'block', marginTop: theme.spacing.md, marginBottom: theme.spacing.xs }}
            >
              <FormattedMessage defaultMessage="Modified output" description="Modified text label in test result" />
            </Typography.Text>
            <div
              css={{
                padding: theme.spacing.sm,
                backgroundColor: '#722ed108',
                borderRadius: theme.borders.borderRadiusMd,
                border: '1px solid #722ed140',
                maxHeight: 120,
                overflowY: 'auto',
                fontSize: theme.typography.fontSizeSm,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                fontFamily: displayModified.isStructured ? 'inherit' : 'monospace',
                lineHeight: 1.5,
              }}
            >
              {displayModified.formatted}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

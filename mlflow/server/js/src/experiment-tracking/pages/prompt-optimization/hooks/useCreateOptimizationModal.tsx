import { useCallback, useMemo, useState } from 'react';
import {
  Alert,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxSectionHeader,
  DialogComboboxTrigger,
  FormUI,
  InfoSmallIcon,
  Modal,
  RHFControlledComponents,
  SegmentedControlButton,
  SegmentedControlGroup,
  Spacer,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useForm, Controller, FormProvider } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import { useMutation, useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { PromptOptimizationApi } from '../api';
import { OptimizerType, type OptimizerTypeValue } from '../types';
import { RegisteredPromptsApi } from '../../prompts/api';
import type { RegisteredPrompt } from '../../prompts/types';
import { listScheduledScorers, type MLflowScorer } from '../../experiment-scorers/api';
import { TRACE_LEVEL_LLM_TEMPLATES, LLM_TEMPLATE } from '../../experiment-scorers/types';
import { useSearchEvaluationDatasets } from '../../experiment-evaluation-datasets/hooks/useSearchEvaluationDatasets';
import { useProviderModelData } from '../../prompts/hooks/useProviderModelData';

interface UseCreateOptimizationModalProps {
  experimentId: string;
  onSuccess?: () => void;
}

interface CreateOptimizationFormData {
  promptName: string;
  promptVersion: string;
  datasetId: string;
  scorerNames: string[]; // Changed to array for multi-select
  optimizerType: OptimizerTypeValue;
  reflectionProvider: string;
  reflectionModel: string;
  maxMetricCalls: string;
  // Distillation-specific fields
  teacherPromptName: string;
  teacherPromptVersion: string;
  studentModelProvider: string;
  studentModelName: string;
}

// Built-in scorers that return numeric or yes/no values (suitable for optimization)
// These are based on trace-level LLM templates that produce scores
const NUMERIC_BUILTIN_SCORERS = [
  LLM_TEMPLATE.COMPLETENESS,
  LLM_TEMPLATE.CORRECTNESS,
  LLM_TEMPLATE.FLUENCY,
  LLM_TEMPLATE.RELEVANCE_TO_QUERY,
  LLM_TEMPLATE.RETRIEVAL_GROUNDEDNESS,
  LLM_TEMPLATE.RETRIEVAL_RELEVANCE,
  LLM_TEMPLATE.RETRIEVAL_SUFFICIENCY,
  LLM_TEMPLATE.SAFETY,
  LLM_TEMPLATE.TOOL_CALL_CORRECTNESS,
  LLM_TEMPLATE.TOOL_CALL_EFFICIENCY,
];

/**
 * Hook for managing the Create Optimization modal.
 */
export const useCreateOptimizationModal = ({ experimentId, onSuccess }: UseCreateOptimizationModalProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isScorerDropdownOpen, setIsScorerDropdownOpen] = useState(false);
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const form = useForm<CreateOptimizationFormData>({
    defaultValues: {
      promptName: '',
      promptVersion: '',
      datasetId: '',
      scorerNames: [], // Changed to array for multi-select
      optimizerType: OptimizerType.GEPA,
      reflectionProvider: '',
      reflectionModel: '',
      maxMetricCalls: '100',
      // Distillation-specific fields
      teacherPromptName: '',
      teacherPromptVersion: '',
      studentModelProvider: '',
      studentModelName: '',
    },
  });

  // Fetch prompts for the experiment
  const { data: promptsData, isLoading: promptsLoading } = useQuery<{ registered_models?: RegisteredPrompt[] }, Error>(
    ['prompts_for_optimization', experimentId],
    {
      queryFn: () => RegisteredPromptsApi.listRegisteredPrompts(undefined, undefined, experimentId),
      enabled: isOpen,
      retry: false,
    },
  );

  // Fetch datasets for the experiment
  const { data: datasetsData, isLoading: datasetsLoading } = useSearchEvaluationDatasets({
    experimentId,
    enabled: isOpen,
  });

  // Fetch scorers for the experiment
  const { data: scorersData, isLoading: scorersLoading } = useQuery<{ scorers?: MLflowScorer[] }, Error>(
    ['scorers_for_optimization', experimentId],
    {
      queryFn: () => listScheduledScorers(experimentId),
      enabled: isOpen,
      retry: false,
    },
  );

  // Watch form fields for validation and dependent queries
  const promptName = form.watch('promptName');
  const promptVersion = form.watch('promptVersion');
  const scorerNames = form.watch('scorerNames');
  const datasetId = form.watch('datasetId');

  // Get prompt versions when a prompt is selected
  const { data: versionsData, isLoading: versionsLoading } = useQuery<
    { model_versions?: { version: string }[] },
    Error
  >(['prompt_versions_for_optimization', promptName], {
    queryFn: () => RegisteredPromptsApi.getPromptVersions(promptName),
    enabled: isOpen && !!promptName,
    retry: false,
  });

  // Provider/model data for reflection model selection
  const selectedReflectionProvider = form.watch('reflectionProvider');
  const handleProviderChange = useCallback(() => {
    form.setValue('reflectionModel', '');
  }, [form]);

  const { providers, providersLoading, models, modelsLoading } = useProviderModelData(
    selectedReflectionProvider,
    handleProviderChange,
  );

  // Provider/model data for student model selection (Distillation)
  const selectedStudentProvider = form.watch('studentModelProvider');
  const handleStudentProviderChange = useCallback(() => {
    form.setValue('studentModelName', '');
  }, [form]);

  const {
    providers: studentProviders,
    providersLoading: studentProvidersLoading,
    models: studentModels,
    modelsLoading: studentModelsLoading,
  } = useProviderModelData(selectedStudentProvider, handleStudentProviderChange);

  // Watch teacher prompt fields for version query
  const teacherPromptName = form.watch('teacherPromptName');

  // Get teacher prompt versions when a teacher prompt is selected
  const { data: teacherVersionsData, isLoading: teacherVersionsLoading } = useQuery<
    { model_versions?: { version: string }[] },
    Error
  >(['teacher_prompt_versions_for_optimization', teacherPromptName], {
    queryFn: () => RegisteredPromptsApi.getPromptVersions(teacherPromptName),
    enabled: isOpen && !!teacherPromptName,
    retry: false,
  });

  const teacherVersions = useMemo(() => teacherVersionsData?.model_versions ?? [], [teacherVersionsData]);

  // Create job mutation
  const createMutation = useMutation({
    mutationFn: PromptOptimizationApi.createJob,
    onSuccess: () => {
      onSuccess?.();
      closeModal();
    },
  });

  const openModal = () => {
    form.reset({
      promptName: '',
      promptVersion: '',
      datasetId: '',
      scorerNames: [], // Reset to empty array
      optimizerType: OptimizerType.GEPA,
      reflectionProvider: '',
      reflectionModel: '',
      maxMetricCalls: '100',
      // Distillation-specific fields
      teacherPromptName: '',
      teacherPromptVersion: '',
      studentModelProvider: '',
      studentModelName: '',
    });
    createMutation.reset();
    setIsOpen(true);
  };

  const closeModal = () => setIsOpen(false);

  const prompts = useMemo(() => promptsData?.registered_models ?? [], [promptsData]);
  const versions = useMemo(() => versionsData?.model_versions ?? [], [versionsData]);

  // Combine experiment scorers with built-in scorers
  const allScorerOptions = useMemo(() => {
    const experimentScorers = (scorersData?.scorers ?? []).map((scorer) => ({
      name: scorer.scorer_name,
      type: 'experiment' as const,
    }));

    const builtInScorers = NUMERIC_BUILTIN_SCORERS.map((template) => ({
      name: template,
      type: 'builtin' as const,
    }));

    // Combine, with built-in scorers first
    return [...builtInScorers, ...experimentScorers];
  }, [scorersData]);

  const optimizerType = form.watch('optimizerType');
  const isGEPA = optimizerType === OptimizerType.GEPA;
  const isDistillation = optimizerType === OptimizerType.DISTILLATION;

  const handleSubmit = form.handleSubmit(async (values) => {
    // Handle Distillation differently
    if (values.optimizerType === OptimizerType.DISTILLATION) {
      // Build teacher prompt URI
      const teacherPromptUri = `prompts:/${values.teacherPromptName}/${values.teacherPromptVersion}`;

      // Build student model config
      const studentModelConfig = {
        provider: values.studentModelProvider,
        model_name: values.studentModelName,
      };

      // Build optimizer config
      const optimizerConfig: Record<string, any> = {};
      if (values.reflectionProvider && values.reflectionModel) {
        optimizerConfig['reflection_model'] = `${values.reflectionProvider}:/${values.reflectionModel}`;
      }

      createMutation.mutate({
        experiment_id: experimentId,
        source_prompt_uri: teacherPromptUri, // Teacher prompt is the source for template
        config: {
          optimizer_type: values.optimizerType,
          teacher_prompt_uri: teacherPromptUri,
          student_model_config_json: JSON.stringify(studentModelConfig),
          optimizer_config_json: Object.keys(optimizerConfig).length > 0 ? JSON.stringify(optimizerConfig) : undefined,
        },
      });
    } else {
      // Build the source prompt URI
      // Format: prompts:/name/version (e.g., prompts:/my_prompt/1)
      const sourcePromptUri = `prompts:/${values.promptName}/${values.promptVersion}`;

      // Build optimizer config
      const optimizerConfig: Record<string, any> = {};
      if (values.reflectionProvider && values.reflectionModel) {
        // Format: provider:/model-name (e.g., openai:/gpt-4.1-mini)
        optimizerConfig['reflection_model'] = `${values.reflectionProvider}:/${values.reflectionModel}`;
      }
      if (isGEPA && values.maxMetricCalls) {
        optimizerConfig['max_metric_calls'] = parseInt(values.maxMetricCalls, 10);
      }

      createMutation.mutate({
        experiment_id: experimentId,
        source_prompt_uri: sourcePromptUri,
        config: {
          optimizer_type: values.optimizerType,
          dataset_id: values.datasetId || undefined,
          scorers: values.scorerNames, // Now uses the array directly
          optimizer_config_json: Object.keys(optimizerConfig).length > 0 ? JSON.stringify(optimizerConfig) : undefined,
        },
      });
    }
  });

  // Watch distillation-specific fields for validation
  const teacherPromptVersion = form.watch('teacherPromptVersion');
  const studentModelProvider = form.watch('studentModelProvider');
  const studentModelName = form.watch('studentModelName');

  // Check if form is valid for submission
  const isFormValid = useMemo(() => {
    // Distillation has different validation
    if (optimizerType === OptimizerType.DISTILLATION) {
      return !!(teacherPromptName && teacherPromptVersion && studentModelProvider && studentModelName);
    }

    if (!promptName || !promptVersion) return false;

    // GEPA always requires both dataset and scorers
    if (optimizerType === OptimizerType.GEPA) {
      if (!datasetId) return false;
      if (!scorerNames || scorerNames.length === 0) return false;
    }

    // MetaPrompt validation:
    // - Zero-shot: no dataset, no scorers - valid
    // - Few-shot without eval: dataset only - valid
    // - Few-shot with eval: dataset + scorers - valid
    // Note: scorers without dataset doesn't make sense, but we'll allow backend to validate that
    return true;
  }, [
    promptName,
    promptVersion,
    scorerNames,
    optimizerType,
    datasetId,
    teacherPromptName,
    teacherPromptVersion,
    studentModelProvider,
    studentModelName,
  ]);

  const CreateOptimizationModal = (
    <FormProvider {...form}>
      <Modal
        componentId="mlflow.prompt-optimization.create-modal"
        visible={isOpen}
        onCancel={closeModal}
        title={
          <FormattedMessage
            defaultMessage="Create New Optimization"
            description="Title for the create optimization modal"
          />
        }
        footer={
          <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
            <button
              type="button"
              onClick={closeModal}
              css={{
                padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.general.borderRadiusBase,
                background: theme.colors.backgroundPrimary,
                cursor: 'pointer',
              }}
            >
              <FormattedMessage defaultMessage="Cancel" description="Cancel button label" />
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!isFormValid || createMutation.isLoading}
              css={{
                padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
                border: 'none',
                borderRadius: theme.general.borderRadiusBase,
                background: isFormValid
                  ? theme.colors.actionPrimaryBackgroundDefault
                  : theme.colors.actionDisabledBackground,
                color: isFormValid ? theme.colors.actionPrimaryTextDefault : theme.colors.actionDisabledText,
                cursor: isFormValid ? 'pointer' : 'not-allowed',
              }}
            >
              {createMutation.isLoading ? (
                <FormattedMessage defaultMessage="Creating..." description="Creating button label" />
              ) : (
                <FormattedMessage
                  defaultMessage="Create optimization"
                  description="Submit button label for create optimization modal"
                />
              )}
            </button>
          </div>
        }
        size="wide"
      >
        {createMutation.error && (
          <>
            <Alert
              componentId="mlflow.prompt-optimization.create-modal.error"
              closable={false}
              message={(createMutation.error as Error).message}
              type="error"
            />
            <Spacer />
          </>
        )}

        {/* Prompt Selection - Hidden for Distillation */}
        {!isDistillation && (
          <>
            <FormUI.Label htmlFor="mlflow.prompt-optimization.create.prompt">
              <FormattedMessage defaultMessage="Prompt" description="Label for prompt selection" />
            </FormUI.Label>
            <Controller
              name="promptName"
              control={form.control}
              rules={{ required: !isDistillation }}
              render={({ field }) => (
                <DialogCombobox
                  componentId="mlflow.prompt-optimization.create.prompt"
                  label={intl.formatMessage({ defaultMessage: 'Prompt', description: 'Label for prompt selection' })}
                  modal={false}
                  value={field.value ? [field.value] : undefined}
                >
                  <DialogComboboxTrigger
                    id="mlflow.prompt-optimization.create.prompt"
                    css={{ width: '100%' }}
                    allowClear
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Select a prompt',
                      description: 'Placeholder for prompt selection',
                    })}
                    withInlineLabel={false}
                    onClear={() => {
                      field.onChange('');
                      form.setValue('promptVersion', '');
                    }}
                  />
                  <DialogComboboxContent loading={promptsLoading} maxHeight={400} matchTriggerWidth>
                    {!promptsLoading && prompts.length > 0 && (
                      <DialogComboboxOptionList>
                        <DialogComboboxOptionListSearch autoFocus>
                          {prompts.map((prompt) => (
                            <DialogComboboxOptionListSelectItem
                              value={prompt.name}
                              key={prompt.name}
                              onChange={(value) => {
                                field.onChange(value);
                                form.setValue('promptVersion', '');
                              }}
                              checked={field.value === prompt.name}
                            >
                              {prompt.name}
                            </DialogComboboxOptionListSelectItem>
                          ))}
                        </DialogComboboxOptionListSearch>
                      </DialogComboboxOptionList>
                    )}
                  </DialogComboboxContent>
                </DialogCombobox>
              )}
            />
            <Spacer size="sm" />

            {/* Prompt Version Selection */}
            <FormUI.Label htmlFor="mlflow.prompt-optimization.create.version">
              <FormattedMessage defaultMessage="Version" description="Label for prompt version selection" />
            </FormUI.Label>
            <Controller
              name="promptVersion"
              control={form.control}
              rules={{ required: !isDistillation }}
              render={({ field }) => (
                <DialogCombobox
                  componentId="mlflow.prompt-optimization.create.version"
                  label={intl.formatMessage({
                    defaultMessage: 'Version',
                    description: 'Label for prompt version selection',
                  })}
                  modal={false}
                  value={field.value ? [field.value] : undefined}
                >
                  <DialogComboboxTrigger
                    id="mlflow.prompt-optimization.create.version"
                    css={{ width: '100%' }}
                    allowClear
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Select a version',
                      description: 'Placeholder for prompt version selection',
                    })}
                    withInlineLabel={false}
                    disabled={!promptName}
                    onClear={() => field.onChange('')}
                  />
                  <DialogComboboxContent loading={versionsLoading} maxHeight={400} matchTriggerWidth>
                    {!versionsLoading && versions.length > 0 && (
                      <DialogComboboxOptionList>
                        <DialogComboboxOptionListSearch autoFocus>
                          {versions.map((version) => (
                            <DialogComboboxOptionListSelectItem
                              value={version.version}
                              key={version.version}
                              onChange={(value) => field.onChange(value)}
                              checked={field.value === version.version}
                            >
                              {intl.formatMessage(
                                {
                                  defaultMessage: 'Version {version}',
                                  description: 'Label for prompt version option',
                                },
                                { version: version.version },
                              )}
                            </DialogComboboxOptionListSelectItem>
                          ))}
                        </DialogComboboxOptionListSearch>
                      </DialogComboboxOptionList>
                    )}
                  </DialogComboboxContent>
                </DialogCombobox>
              )}
            />
            <Spacer />
          </>
        )}

        {/* Optimizer Type Selection */}
        <FormUI.Label>
          <FormattedMessage defaultMessage="Optimizer" description="Label for optimizer type selection" />
        </FormUI.Label>
        <Controller
          control={form.control}
          name="optimizerType"
          render={({ field }) => (
            <SegmentedControlGroup
              name="optimizerType"
              componentId="mlflow.prompt-optimization.create.optimizer-type"
              value={field.value}
              onChange={field.onChange}
            >
              <SegmentedControlButton value={OptimizerType.GEPA}>
                <FormattedMessage defaultMessage="GEPA" description="Label for GEPA optimizer type" />
              </SegmentedControlButton>
              <SegmentedControlButton value={OptimizerType.METAPROMPT}>
                <FormattedMessage defaultMessage="MetaPrompt" description="Label for MetaPrompt optimizer type" />
              </SegmentedControlButton>
              <SegmentedControlButton value={OptimizerType.DISTILLATION}>
                <FormattedMessage defaultMessage="Distillation" description="Label for Distillation optimizer type" />
              </SegmentedControlButton>
            </SegmentedControlGroup>
          )}
        />
        {/* Info box explaining the selected optimizer */}
        <div
          css={{
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.general.borderRadiusBase,
            padding: theme.spacing.md,
            marginTop: theme.spacing.sm,
            display: 'flex',
            gap: theme.spacing.sm,
          }}
        >
          <InfoSmallIcon css={{ color: theme.colors.textSecondary, flexShrink: 0, marginTop: 2 }} />
          <div>
            {isGEPA ? (
              <>
                <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                  <FormattedMessage defaultMessage="GEPA (Genetic Prompt Algorithm)" description="GEPA title" />
                </Typography.Text>
                <Typography.Text color="secondary">
                  <FormattedMessage
                    defaultMessage="Uses evolutionary algorithms to iteratively improve prompts. Requires both a dataset and scorer(s) for evaluation-driven optimization."
                    description="Description for GEPA optimizer"
                  />
                </Typography.Text>
              </>
            ) : isDistillation ? (
              <>
                <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                  <FormattedMessage defaultMessage="Distillation" description="Distillation title" />
                </Typography.Text>
                <Typography.Text color="secondary">
                  <FormattedMessage
                    defaultMessage="Distill knowledge from a teacher prompt (with existing traces) to a smaller, more efficient student model. The student prompt is auto-created with the same template but different model configuration."
                    description="Description for Distillation optimizer"
                  />
                </Typography.Text>
              </>
            ) : (
              <>
                <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                  <FormattedMessage defaultMessage="MetaPrompt" description="MetaPrompt title" />
                </Typography.Text>
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                  <FormattedMessage
                    defaultMessage="Uses an LLM to generate improved prompts. Supports multiple modes:"
                    description="Description for MetaPrompt optimizer"
                  />
                </Typography.Text>
                <ul css={{ margin: 0, paddingLeft: theme.spacing.lg, color: theme.colors.textSecondary }}>
                  <li>
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="Zero-shot: No dataset or scorers needed"
                        description="Zero-shot mode description"
                      />
                    </Typography.Text>
                  </li>
                  <li>
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="Few-shot: Add a dataset for example-driven improvement"
                        description="Few-shot mode description"
                      />
                    </Typography.Text>
                  </li>
                  <li>
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="Few-shot + evaluation: Add dataset and scorer(s) for validated improvements"
                        description="Few-shot with eval mode description"
                      />
                    </Typography.Text>
                  </li>
                </ul>
              </>
            )}
          </div>
        </div>
        <Spacer />

        {/* Distillation-specific form fields */}
        {isDistillation && (
          <>
            {/* Teacher Prompt Selection */}
            <FormUI.Label htmlFor="mlflow.prompt-optimization.create.teacher-prompt">
              <FormattedMessage defaultMessage="Teacher Prompt" description="Label for teacher prompt selection" />
            </FormUI.Label>
            <Controller
              name="teacherPromptName"
              control={form.control}
              rules={{ required: isDistillation }}
              render={({ field }) => (
                <DialogCombobox
                  componentId="mlflow.prompt-optimization.create.teacher-prompt"
                  label={intl.formatMessage({
                    defaultMessage: 'Teacher Prompt',
                    description: 'Label for teacher prompt selection',
                  })}
                  modal={false}
                  value={field.value ? [field.value] : undefined}
                >
                  <DialogComboboxTrigger
                    id="mlflow.prompt-optimization.create.teacher-prompt"
                    css={{ width: '100%' }}
                    allowClear
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Select a teacher prompt',
                      description: 'Placeholder for teacher prompt selection',
                    })}
                    withInlineLabel={false}
                    onClear={() => {
                      field.onChange('');
                      form.setValue('teacherPromptVersion', '');
                    }}
                  />
                  <DialogComboboxContent loading={promptsLoading} maxHeight={400} matchTriggerWidth>
                    {!promptsLoading && prompts.length > 0 && (
                      <DialogComboboxOptionList>
                        <DialogComboboxOptionListSearch autoFocus>
                          {prompts.map((prompt) => (
                            <DialogComboboxOptionListSelectItem
                              value={prompt.name}
                              key={prompt.name}
                              onChange={(value) => {
                                field.onChange(value);
                                form.setValue('teacherPromptVersion', '');
                              }}
                              checked={field.value === prompt.name}
                            >
                              {prompt.name}
                            </DialogComboboxOptionListSelectItem>
                          ))}
                        </DialogComboboxOptionListSearch>
                      </DialogComboboxOptionList>
                    )}
                  </DialogComboboxContent>
                </DialogCombobox>
              )}
            />
            <Spacer size="sm" />

            {/* Teacher Prompt Version Selection */}
            <FormUI.Label htmlFor="mlflow.prompt-optimization.create.teacher-version">
              <FormattedMessage
                defaultMessage="Teacher Prompt Version"
                description="Label for teacher prompt version selection"
              />
            </FormUI.Label>
            <Controller
              name="teacherPromptVersion"
              control={form.control}
              rules={{ required: isDistillation }}
              render={({ field }) => (
                <DialogCombobox
                  componentId="mlflow.prompt-optimization.create.teacher-version"
                  label={intl.formatMessage({
                    defaultMessage: 'Version',
                    description: 'Label for teacher prompt version selection',
                  })}
                  modal={false}
                  value={field.value ? [field.value] : undefined}
                >
                  <DialogComboboxTrigger
                    id="mlflow.prompt-optimization.create.teacher-version"
                    css={{ width: '100%' }}
                    allowClear
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Select a version',
                      description: 'Placeholder for teacher prompt version selection',
                    })}
                    withInlineLabel={false}
                    disabled={!teacherPromptName}
                    onClear={() => field.onChange('')}
                  />
                  <DialogComboboxContent loading={teacherVersionsLoading} maxHeight={400} matchTriggerWidth>
                    {!teacherVersionsLoading && teacherVersions.length > 0 && (
                      <DialogComboboxOptionList>
                        <DialogComboboxOptionListSearch autoFocus>
                          {teacherVersions.map((version) => (
                            <DialogComboboxOptionListSelectItem
                              value={version.version}
                              key={version.version}
                              onChange={(value) => field.onChange(value)}
                              checked={field.value === version.version}
                            >
                              {intl.formatMessage(
                                {
                                  defaultMessage: 'Version {version}',
                                  description: 'Label for teacher prompt version option',
                                },
                                { version: version.version },
                              )}
                            </DialogComboboxOptionListSelectItem>
                          ))}
                        </DialogComboboxOptionListSearch>
                      </DialogComboboxOptionList>
                    )}
                  </DialogComboboxContent>
                </DialogCombobox>
              )}
            />
            <Spacer />

            {/* Student Model Configuration */}
            <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
              <FormattedMessage
                defaultMessage="Student Model Configuration"
                description="Section header for student model configuration"
              />
            </Typography.Title>
            <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
              <FormattedMessage
                defaultMessage="The student prompt will be auto-created with the same template as the teacher but configured for this model."
                description="Help text for student model configuration"
              />
            </Typography.Text>

            {/* Student Model Provider */}
            <FormUI.Label htmlFor="mlflow.prompt-optimization.create.student-provider">
              <FormattedMessage defaultMessage="Student Model Provider" description="Label for student provider" />
            </FormUI.Label>
            <Controller
              name="studentModelProvider"
              control={form.control}
              rules={{ required: isDistillation }}
              render={({ field }) => (
                <DialogCombobox
                  componentId="mlflow.prompt-optimization.create.student-provider"
                  label={intl.formatMessage({
                    defaultMessage: 'Provider',
                    description: 'Label for student provider',
                  })}
                  modal={false}
                  value={field.value ? [field.value] : undefined}
                >
                  <DialogComboboxTrigger
                    id="mlflow.prompt-optimization.create.student-provider"
                    css={{ width: '100%' }}
                    allowClear
                    placeholder={intl.formatMessage({
                      defaultMessage: 'e.g., openai, anthropic',
                      description: 'Placeholder for student provider',
                    })}
                    withInlineLabel={false}
                    onClear={() => {
                      field.onChange('');
                      form.setValue('studentModelName', '');
                    }}
                  />
                  <DialogComboboxContent loading={studentProvidersLoading} maxHeight={400} matchTriggerWidth>
                    {!studentProvidersLoading && studentProviders && (
                      <DialogComboboxOptionList>
                        <DialogComboboxOptionListSearch autoFocus>
                          {studentProviders.map((provider) => (
                            <DialogComboboxOptionListSelectItem
                              value={provider}
                              key={provider}
                              onChange={(value) => field.onChange(value)}
                              checked={field.value === provider}
                            >
                              {provider}
                            </DialogComboboxOptionListSelectItem>
                          ))}
                        </DialogComboboxOptionListSearch>
                      </DialogComboboxOptionList>
                    )}
                  </DialogComboboxContent>
                </DialogCombobox>
              )}
            />
            <Spacer size="sm" />

            {/* Student Model Name */}
            <FormUI.Label htmlFor="mlflow.prompt-optimization.create.student-model">
              <FormattedMessage defaultMessage="Student Model" description="Label for student model" />
            </FormUI.Label>
            <Controller
              name="studentModelName"
              control={form.control}
              rules={{ required: isDistillation }}
              render={({ field }) => (
                <DialogCombobox
                  componentId="mlflow.prompt-optimization.create.student-model"
                  label={intl.formatMessage({
                    defaultMessage: 'Model',
                    description: 'Label for student model',
                  })}
                  modal={false}
                  value={field.value ? [field.value] : undefined}
                >
                  <DialogComboboxTrigger
                    id="mlflow.prompt-optimization.create.student-model"
                    css={{ width: '100%' }}
                    allowClear
                    placeholder={intl.formatMessage({
                      defaultMessage: 'e.g., gpt-4o-mini, claude-3-haiku',
                      description: 'Placeholder for student model',
                    })}
                    withInlineLabel={false}
                    disabled={!selectedStudentProvider}
                    onClear={() => field.onChange('')}
                  />
                  <DialogComboboxContent loading={studentModelsLoading} maxHeight={400} matchTriggerWidth>
                    {!studentModelsLoading && studentModels && (
                      <DialogComboboxOptionList>
                        <DialogComboboxOptionListSearch autoFocus>
                          {studentModels.map((model) => (
                            <DialogComboboxOptionListSelectItem
                              value={model.model}
                              key={model.model}
                              onChange={(value) => field.onChange(value)}
                              checked={field.value === model.model}
                            >
                              {model.model}
                            </DialogComboboxOptionListSelectItem>
                          ))}
                        </DialogComboboxOptionListSearch>
                      </DialogComboboxOptionList>
                    )}
                  </DialogComboboxContent>
                </DialogCombobox>
              )}
            />
            <Spacer />
          </>
        )}

        {/* Dataset Selection - Hidden for Distillation */}
        {!isDistillation && (
          <>
            <FormUI.Label htmlFor="mlflow.prompt-optimization.create.dataset">
              <FormattedMessage defaultMessage="Dataset" description="Label for dataset selection" />
              {!isGEPA && (
                <Typography.Text color="secondary" css={{ marginLeft: theme.spacing.xs }}>
                  <FormattedMessage defaultMessage="(optional)" description="Optional label" />
                </Typography.Text>
              )}
            </FormUI.Label>
            <Controller
              name="datasetId"
              control={form.control}
              rules={{ required: isGEPA }}
              render={({ field }) => {
                // Find the selected dataset to display its name
                const selectedDataset = datasetsData.find((d) => d.dataset_id === field.value);
                const displayValue = selectedDataset ? selectedDataset.name || selectedDataset.dataset_id : undefined;

                return (
                  <DialogCombobox
                    componentId="mlflow.prompt-optimization.create.dataset"
                    label={intl.formatMessage({
                      defaultMessage: 'Dataset',
                      description: 'Label for dataset selection',
                    })}
                    modal={false}
                    value={displayValue ? [displayValue] : undefined}
                  >
                    <DialogComboboxTrigger
                      id="mlflow.prompt-optimization.create.dataset"
                      css={{ width: '100%' }}
                      allowClear
                      placeholder={intl.formatMessage({
                        defaultMessage: 'Select a dataset',
                        description: 'Placeholder for dataset selection',
                      })}
                      withInlineLabel={false}
                      onClear={() => field.onChange('')}
                    />
                    <DialogComboboxContent loading={datasetsLoading} maxHeight={400} matchTriggerWidth>
                      {!datasetsLoading && datasetsData.length > 0 && (
                        <DialogComboboxOptionList>
                          <DialogComboboxOptionListSearch autoFocus>
                            {datasetsData.map((dataset) => (
                              <DialogComboboxOptionListSelectItem
                                value={dataset.dataset_id}
                                key={dataset.dataset_id}
                                onChange={(value) => field.onChange(value)}
                                checked={field.value === dataset.dataset_id}
                              >
                                {dataset.name || dataset.dataset_id}
                              </DialogComboboxOptionListSelectItem>
                            ))}
                          </DialogComboboxOptionListSearch>
                        </DialogComboboxOptionList>
                      )}
                    </DialogComboboxContent>
                  </DialogCombobox>
                );
              }}
            />
            <Spacer />
          </>
        )}

        {/* Scorer Selection - Multi-select with checkbox items - Hidden for Distillation */}
        {!isDistillation && (
          <>
            <FormUI.Label htmlFor="mlflow.prompt-optimization.create.scorer">
              <FormattedMessage defaultMessage="Scorer(s)" description="Label for scorer selection" />
              {!isGEPA && (
                <Typography.Text color="secondary" css={{ marginLeft: theme.spacing.xs }}>
                  <FormattedMessage defaultMessage="(optional)" description="Optional label" />
                </Typography.Text>
              )}
            </FormUI.Label>
            <Controller
              name="scorerNames"
              control={form.control}
              rules={{ required: isGEPA }}
              render={({ field }) => (
                <>
                  <DialogCombobox
                    componentId="mlflow.prompt-optimization.create.scorer"
                    label={intl.formatMessage({ defaultMessage: 'Scorers', description: 'Label for scorer selection' })}
                    open={isScorerDropdownOpen}
                    onOpenChange={setIsScorerDropdownOpen}
                    multiSelect
                  >
                    <DialogComboboxTrigger
                      id="mlflow.prompt-optimization.create.scorer"
                      css={{ width: '100%' }}
                      allowClear={false}
                      placeholder={intl.formatMessage({
                        defaultMessage: 'Click to select scorers',
                        description: 'Placeholder for scorer selection',
                      })}
                      withInlineLabel={false}
                    />
                    <DialogComboboxContent loading={scorersLoading} maxHeight={400} matchTriggerWidth>
                      {!scorersLoading && allScorerOptions.length > 0 && (
                        <DialogComboboxOptionList>
                          <DialogComboboxSectionHeader>
                            <FormattedMessage
                              defaultMessage="Built-in Scorers"
                              description="Built-in scorers section"
                            />
                          </DialogComboboxSectionHeader>
                          {allScorerOptions
                            .filter((s) => s.type === 'builtin')
                            .map((scorer) => (
                              <DialogComboboxOptionListCheckboxItem
                                value={scorer.name}
                                key={`builtin-${scorer.name}`}
                                onChange={() => {
                                  const isSelected = field.value.includes(scorer.name);
                                  if (isSelected) {
                                    field.onChange(field.value.filter((v: string) => v !== scorer.name));
                                  } else {
                                    field.onChange([...field.value, scorer.name]);
                                  }
                                }}
                                checked={field.value.includes(scorer.name)}
                              >
                                {scorer.name}
                              </DialogComboboxOptionListCheckboxItem>
                            ))}
                          {allScorerOptions.some((s) => s.type === 'experiment') && (
                            <>
                              <Spacer size="xs" />
                              <DialogComboboxSectionHeader>
                                <FormattedMessage
                                  defaultMessage="Experiment Scorers"
                                  description="Experiment scorers section"
                                />
                              </DialogComboboxSectionHeader>
                              {allScorerOptions
                                .filter((s) => s.type === 'experiment')
                                .map((scorer) => (
                                  <DialogComboboxOptionListCheckboxItem
                                    value={scorer.name}
                                    key={`experiment-${scorer.name}`}
                                    onChange={() => {
                                      const isSelected = field.value.includes(scorer.name);
                                      if (isSelected) {
                                        field.onChange(field.value.filter((v: string) => v !== scorer.name));
                                      } else {
                                        field.onChange([...field.value, scorer.name]);
                                      }
                                    }}
                                    checked={field.value.includes(scorer.name)}
                                  >
                                    {scorer.name}
                                  </DialogComboboxOptionListCheckboxItem>
                                ))}
                            </>
                          )}
                        </DialogComboboxOptionList>
                      )}
                    </DialogComboboxContent>
                  </DialogCombobox>

                  {/* Display selected scorers as removable tags */}
                  {field.value.length > 0 && (
                    <div
                      css={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: theme.spacing.xs,
                        marginTop: theme.spacing.sm,
                      }}
                    >
                      {field.value.map((scorerName: string) => {
                        const scorerOption = allScorerOptions.find((s) => s.name === scorerName);
                        const isBuiltin = scorerOption?.type === 'builtin';
                        return (
                          <Tag
                            key={scorerName}
                            componentId={`mlflow.prompt-optimization.create.scorer.selected-${scorerName}`}
                            color={isBuiltin ? 'turquoise' : 'purple'}
                            closable
                            onClose={() => {
                              field.onChange(field.value.filter((v: string) => v !== scorerName));
                            }}
                          >
                            {scorerName}
                          </Tag>
                        );
                      })}
                    </div>
                  )}
                </>
              )}
            />
            <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs, display: 'block' }}>
              <FormattedMessage
                defaultMessage="Select built-in scorers (turquoise) or experiment-specific scorers (purple). Click the X to remove."
                description="Help text for scorer selection"
              />
            </Typography.Text>
            <Spacer />
          </>
        )}

        {/* Optimizer Configuration Section */}
        <div
          css={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.general.borderRadiusBase,
            padding: theme.spacing.md,
          }}
        >
          <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Optimizer Configuration"
              description="Section header for optimizer configuration"
            />
          </Typography.Title>

          {/* Reflection Model Provider */}
          <FormUI.Label htmlFor="mlflow.prompt-optimization.create.reflection-provider">
            <FormattedMessage defaultMessage="Reflection Model Provider" description="Label for reflection provider" />
          </FormUI.Label>
          <Controller
            name="reflectionProvider"
            control={form.control}
            render={({ field }) => (
              <DialogCombobox
                componentId="mlflow.prompt-optimization.create.reflection-provider"
                label={intl.formatMessage({
                  defaultMessage: 'Provider',
                  description: 'Label for reflection provider',
                })}
                modal={false}
                value={field.value ? [field.value] : undefined}
              >
                <DialogComboboxTrigger
                  id="mlflow.prompt-optimization.create.reflection-provider"
                  css={{ width: '100%' }}
                  allowClear
                  placeholder={intl.formatMessage({
                    defaultMessage: 'e.g., openai, anthropic',
                    description: 'Placeholder for reflection provider',
                  })}
                  withInlineLabel={false}
                  onClear={() => {
                    field.onChange('');
                    form.setValue('reflectionModel', '');
                  }}
                />
                <DialogComboboxContent loading={providersLoading} maxHeight={400} matchTriggerWidth>
                  {!providersLoading && providers && (
                    <DialogComboboxOptionList>
                      <DialogComboboxOptionListSearch autoFocus>
                        {providers.map((provider) => (
                          <DialogComboboxOptionListSelectItem
                            value={provider}
                            key={provider}
                            onChange={(value) => field.onChange(value)}
                            checked={field.value === provider}
                          >
                            {provider}
                          </DialogComboboxOptionListSelectItem>
                        ))}
                      </DialogComboboxOptionListSearch>
                    </DialogComboboxOptionList>
                  )}
                </DialogComboboxContent>
              </DialogCombobox>
            )}
          />
          <Spacer size="sm" />

          {/* Reflection Model Name */}
          <FormUI.Label htmlFor="mlflow.prompt-optimization.create.reflection-model">
            <FormattedMessage defaultMessage="Reflection Model" description="Label for reflection model" />
          </FormUI.Label>
          <Controller
            name="reflectionModel"
            control={form.control}
            render={({ field }) => (
              <DialogCombobox
                componentId="mlflow.prompt-optimization.create.reflection-model"
                label={intl.formatMessage({
                  defaultMessage: 'Model',
                  description: 'Label for reflection model',
                })}
                modal={false}
                value={field.value ? [field.value] : undefined}
              >
                <DialogComboboxTrigger
                  id="mlflow.prompt-optimization.create.reflection-model"
                  css={{ width: '100%' }}
                  allowClear
                  placeholder={intl.formatMessage({
                    defaultMessage: 'e.g., gpt-4o, claude-3-5-sonnet',
                    description: 'Placeholder for reflection model',
                  })}
                  withInlineLabel={false}
                  disabled={!selectedReflectionProvider}
                  onClear={() => field.onChange('')}
                />
                <DialogComboboxContent loading={modelsLoading} maxHeight={400} matchTriggerWidth>
                  {!modelsLoading && models && (
                    <DialogComboboxOptionList>
                      <DialogComboboxOptionListSearch autoFocus>
                        {models.map((model) => (
                          <DialogComboboxOptionListSelectItem
                            value={model.model}
                            key={model.model}
                            onChange={(value) => field.onChange(value)}
                            checked={field.value === model.model}
                          >
                            {model.model}
                          </DialogComboboxOptionListSelectItem>
                        ))}
                      </DialogComboboxOptionListSearch>
                    </DialogComboboxOptionList>
                  )}
                </DialogComboboxContent>
              </DialogCombobox>
            )}
          />

          {/* Max Metric Calls (GEPA only) */}
          {isGEPA && (
            <>
              <Spacer size="sm" />
              <FormUI.Label htmlFor="mlflow.prompt-optimization.create.max-metric-calls">
                <FormattedMessage defaultMessage="Max Metric Calls" description="Label for max metric calls input" />
              </FormUI.Label>
              <RHFControlledComponents.Input
                control={form.control}
                id="mlflow.prompt-optimization.create.max-metric-calls"
                componentId="mlflow.prompt-optimization.create.max-metric-calls"
                name="maxMetricCalls"
                type="number"
                placeholder={intl.formatMessage({
                  defaultMessage: 'Default: 100',
                  description: 'Placeholder for max metric calls',
                })}
              />
              <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs, display: 'block' }}>
                <FormattedMessage
                  defaultMessage="Maximum number of evaluation calls during optimization."
                  description="Help text for max metric calls"
                />
              </Typography.Text>
            </>
          )}
        </div>
      </Modal>
    </FormProvider>
  );

  return {
    CreateOptimizationModal,
    openModal,
    closeModal,
  };
};

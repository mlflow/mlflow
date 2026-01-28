import { useCallback, useMemo, useState } from 'react';
import {
  Alert,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  Modal,
  RHFControlledComponents,
  SegmentedControlButton,
  SegmentedControlGroup,
  Spacer,
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
  scorerName: string;
  optimizerType: OptimizerTypeValue;
  reflectionProvider: string;
  reflectionModel: string;
  maxMetricCalls: string;
}

/**
 * Hook for managing the Create Optimization modal.
 */
export const useCreateOptimizationModal = ({ experimentId, onSuccess }: UseCreateOptimizationModalProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const form = useForm<CreateOptimizationFormData>({
    defaultValues: {
      promptName: '',
      promptVersion: '',
      datasetId: '',
      scorerName: '',
      optimizerType: OptimizerType.GEPA,
      reflectionProvider: '',
      reflectionModel: '',
      maxMetricCalls: '100',
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
  const scorerName = form.watch('scorerName');
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
      scorerName: '',
      optimizerType: OptimizerType.GEPA,
      reflectionProvider: '',
      reflectionModel: '',
      maxMetricCalls: '100',
    });
    createMutation.reset();
    setIsOpen(true);
  };

  const closeModal = () => setIsOpen(false);

  const prompts = useMemo(() => promptsData?.registered_models ?? [], [promptsData]);
  const versions = useMemo(() => versionsData?.model_versions ?? [], [versionsData]);
  const scorers = useMemo(() => scorersData?.scorers ?? [], [scorersData]);
  const optimizerType = form.watch('optimizerType');
  const isGEPA = optimizerType === OptimizerType.GEPA;

  const handleSubmit = form.handleSubmit(async (values) => {
    // Build the source prompt URI
    const sourcePromptUri = `models:/${values.promptName}/${values.promptVersion}`;

    // Build optimizer config
    const optimizerConfig: Record<string, any> = {};
    if (values.reflectionProvider && values.reflectionModel) {
      optimizerConfig['reflection_model'] = `${values.reflectionProvider}/${values.reflectionModel}`;
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
        scorers: [values.scorerName],
        optimizer_config_json: Object.keys(optimizerConfig).length > 0 ? JSON.stringify(optimizerConfig) : undefined,
      },
    });
  });

  // Check if form is valid for submission
  const isFormValid = useMemo(() => {
    if (!promptName || !promptVersion) return false;
    if (!scorerName) return false;
    // GEPA requires a dataset
    if (optimizerType === OptimizerType.GEPA && !datasetId) return false;
    return true;
  }, [promptName, promptVersion, scorerName, optimizerType, datasetId]);

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

        {/* Prompt Selection */}
        <FormUI.Label htmlFor="mlflow.prompt-optimization.create.prompt">
          <FormattedMessage defaultMessage="Prompt" description="Label for prompt selection" />
        </FormUI.Label>
        <Controller
          name="promptName"
          control={form.control}
          rules={{ required: true }}
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
          rules={{ required: true }}
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
            </SegmentedControlGroup>
          )}
        />
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs, display: 'block' }}>
          {isGEPA ? (
            <FormattedMessage
              defaultMessage="GEPA uses evolutionary algorithms to iteratively improve prompts based on evaluation data."
              description="Description for GEPA optimizer"
            />
          ) : (
            <FormattedMessage
              defaultMessage="MetaPrompt uses an LLM to generate improved prompts without requiring evaluation data."
              description="Description for MetaPrompt optimizer"
            />
          )}
        </Typography.Text>
        <Spacer />

        {/* Dataset Selection */}
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
          render={({ field }) => (
            <DialogCombobox
              componentId="mlflow.prompt-optimization.create.dataset"
              label={intl.formatMessage({ defaultMessage: 'Dataset', description: 'Label for dataset selection' })}
              modal={false}
              value={field.value ? [field.value] : undefined}
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
          )}
        />
        <Spacer />

        {/* Scorer Selection */}
        <FormUI.Label htmlFor="mlflow.prompt-optimization.create.scorer">
          <FormattedMessage defaultMessage="Scorer" description="Label for scorer selection" />
        </FormUI.Label>
        <Controller
          name="scorerName"
          control={form.control}
          rules={{ required: true }}
          render={({ field }) => (
            <DialogCombobox
              componentId="mlflow.prompt-optimization.create.scorer"
              label={intl.formatMessage({ defaultMessage: 'Scorer', description: 'Label for scorer selection' })}
              modal={false}
              value={field.value ? [field.value] : undefined}
            >
              <DialogComboboxTrigger
                id="mlflow.prompt-optimization.create.scorer"
                css={{ width: '100%' }}
                allowClear
                placeholder={intl.formatMessage({
                  defaultMessage: 'Select a scorer',
                  description: 'Placeholder for scorer selection',
                })}
                withInlineLabel={false}
                onClear={() => field.onChange('')}
              />
              <DialogComboboxContent loading={scorersLoading} maxHeight={400} matchTriggerWidth>
                {!scorersLoading && scorers.length > 0 && (
                  <DialogComboboxOptionList>
                    <DialogComboboxOptionListSearch autoFocus>
                      {scorers.map((scorer) => (
                        <DialogComboboxOptionListSelectItem
                          value={scorer.scorer_name}
                          key={scorer.scorer_id}
                          onChange={(value) => field.onChange(value)}
                          checked={field.value === scorer.scorer_name}
                        >
                          {scorer.scorer_name}
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

import {
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxHintRow,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  InfoSmallIcon,
  Input,
  Modal,
  PlusIcon,
  Spinner,
  LegacyTooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { sortBy, compact } from 'lodash';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDispatch, useSelector } from 'react-redux';
import Utils from '../../../common/utils/Utils';
import type { ThunkDispatch } from '../../../redux-types';
import { createPromptLabRunApi } from '../../actions';
import type { ModelGatewayReduxState } from '../../reducers/ModelGatewayReducer';
import type { ModelGatewayResponseType } from '../../sdk/ModelGatewayService';
import { ModelGatewayService } from '../../sdk/ModelGatewayService';
import { ModelGatewayRouteTask } from '../../sdk/MlflowEnums';
import { generateRandomRunName, getDuplicatedRunName } from '../../utils/RunNameUtils';
import { useExperimentIds } from '../experiment-page/hooks/useExperimentIds';
import {
  compilePromptInputText,
  extractEvaluationPrerequisitesForRun,
  extractRequiredInputParamsForRun,
} from '../prompt-engineering/PromptEngineering.utils';
import { EvaluationCreatePromptParameters } from './EvaluationCreatePromptParameters';
import { usePromptEvaluationInputValues } from './hooks/usePromptEvaluationInputValues';
import { usePromptEvaluationParameters } from './hooks/usePromptEvaluationParameters';
import { usePromptEvaluationPromptTemplateValue } from './hooks/usePromptEvaluationPromptTemplateValue';
import { EvaluationCreateRunPromptTemplateErrors } from './components/EvaluationCreateRunPromptTemplateErrors';
import type { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { EvaluationCreatePromptRunModalExamples } from './EvaluationCreatePromptRunModalExamples';
import { EvaluationCreatePromptRunOutput } from './components/EvaluationCreatePromptRunOutput';
import { useExperimentPageViewMode } from '../experiment-page/hooks/useExperimentPageViewMode';
import { searchAllPromptLabAvailableEndpoints } from '../../actions/PromptEngineeringActions';
import { getPromptEngineeringErrorMessage } from './utils/PromptEngineeringErrorUtils';

const { TextArea } = Input;
type Props = {
  isOpen: boolean;
  closeModal: () => void;
  runBeingDuplicated: RunRowType | null;
  visibleRuns?: RunRowType[];
  refreshRuns: (() => Promise<never[]>) | (() => Promise<any> | null) | (() => void);
};

export const EvaluationCreatePromptRunModal = ({
  isOpen,
  closeModal,
  runBeingDuplicated,
  visibleRuns = [],
  refreshRuns,
}: Props): JSX.Element => {
  const [experimentId] = useExperimentIds();
  const { theme } = useDesignSystemTheme();
  const { parameters, updateParameter } = usePromptEvaluationParameters();
  const [, setViewMode] = useExperimentPageViewMode();

  const [selectedModel, updateSelectedModel] = useState('');
  const [newRunName, setNewRunName] = useState('');
  const [isCreatingRun, setIsCreatingRun] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [lastEvaluationError, setLastEvaluationError] = useState<string | null>(null);
  const [evaluationOutput, setEvaluationOutput] = useState('');
  const [evaluationMetadata, setEvaluationMetadata] = useState<Partial<ModelGatewayResponseType['metadata']>>({});
  const [outputDirty, setOutputDirty] = useState(false);
  const [isViewExamplesModalOpen, setViewExamplesModalOpen] = useState(false);
  const cancelTokenRef = useRef<string | null>(null);

  const dispatch = useDispatch<ThunkDispatch>();

  useEffect(() => {
    dispatch(searchAllPromptLabAvailableEndpoints()).catch((e) => {
      Utils.logErrorAndNotifyUser(e?.message || e);
    });
  }, [dispatch]);

  const intl = useIntl();

  const {
    updateInputVariables,
    inputVariables,
    inputVariableValues,
    updateInputVariableValue,
    inputVariableNameViolations,
    clearInputVariableValues,
  } = usePromptEvaluationInputValues();

  const { handleAddVariableToTemplate, savePromptTemplateInputRef, promptTemplate, updatePromptTemplate } =
    usePromptEvaluationPromptTemplateValue();

  useEffect(() => {
    if (isOpen && !runBeingDuplicated) {
      setNewRunName(generateRandomRunName());
    }
  }, [isOpen, runBeingDuplicated]);

  useEffect(() => {
    updateInputVariables(promptTemplate);
  }, [promptTemplate, updateInputVariables]);

  /**
   * If a run duplication is detected, pre-fill the values
   */
  useEffect(() => {
    if (runBeingDuplicated) {
      const {
        promptTemplate: duplicatedPromptTemplate,
        routeName: duplicatedRouteName,
        parameters: duplicatedParameters,
      } = extractEvaluationPrerequisitesForRun(runBeingDuplicated);

      extractRequiredInputParamsForRun(runBeingDuplicated);
      if (duplicatedPromptTemplate) {
        updatePromptTemplate(duplicatedPromptTemplate);
      }
      if (duplicatedParameters.temperature) {
        updateParameter('temperature', duplicatedParameters.temperature);
      }
      if (duplicatedParameters.max_tokens) {
        updateParameter('max_tokens', duplicatedParameters.max_tokens);
      }
      if (duplicatedRouteName) {
        updateSelectedModel(duplicatedRouteName);
      }
      setEvaluationOutput('');
      setOutputDirty(false);
      const duplicatedRunName = getDuplicatedRunName(
        runBeingDuplicated.runName,
        compact(visibleRuns.map(({ runName }) => runName)),
      );
      setNewRunName(duplicatedRunName);
      clearInputVariableValues();
    }
  }, [runBeingDuplicated, clearInputVariableValues, updateParameter, updatePromptTemplate, visibleRuns]);

  const modelRoutesUnified = useSelector(
    ({ modelGateway }: { modelGateway: ModelGatewayReduxState }) => modelGateway.modelGatewayRoutes,
  );

  // In the next version, routes are already filtered
  const supportedModelRouteListUnified = useMemo(
    () => sortBy(Object.values(modelRoutesUnified), 'name'),
    // .filter(
    //   (x) => x.name.includes('ann-chat') || x.name.includes('bedrock-ai21labs'),
    // )
    [modelRoutesUnified],
  );

  // Determines if model gateway routes are being loaded
  const modelRoutesLoading = useSelector(
    ({ modelGateway }: { modelGateway: ModelGatewayReduxState }) => modelGateway.modelGatewayRoutesLoading.loading,
  );
  useEffect(() => {
    if (evaluationOutput) {
      setOutputDirty(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputVariableValues, promptTemplate, parameters, selectedModel]);

  const onHandleSubmit = () => {
    setIsCreatingRun(true);
    const modelRouteName = modelRoutesUnified[selectedModel]?.name;
    const modelParameters = { ...parameters, route_type: modelRoutesUnified[selectedModel]?.type };

    const modelInput = compilePromptInputText(promptTemplate, inputVariableValues);
    dispatch(
      createPromptLabRunApi({
        experimentId,
        promptTemplate,
        modelInput,
        modelParameters,
        modelRouteName,
        promptParameters: inputVariableValues,
        modelOutput: evaluationOutput,
        runName: newRunName,
        modelOutputParameters: evaluationMetadata,
      }),
    )
      .then(() => {
        refreshRuns();
        closeModal();
        setIsCreatingRun(false);
        setViewMode('ARTIFACT');
      })
      .catch((e) => {
        Utils.logErrorAndNotifyUser(e?.message || e);
        // NB: Not using .finally() due to issues with promise implementation in the Jest
        setIsCreatingRun(false);
      });
  };

  const handleEvaluate = useCallback(() => {
    const modelRoute = modelRoutesUnified[selectedModel];
    const cancelToken = Math.random().toString(36);
    cancelTokenRef.current = cancelToken;
    if (!modelRoute) {
      // Should never happen if the model is selected
      throw new Error('No model route found!');
    }
    setLastEvaluationError(null);
    setIsEvaluating(true);
    const inputText = compilePromptInputText(promptTemplate, inputVariableValues);
    const startTime = performance.now();
    ModelGatewayService.queryModelGatewayRoute(modelRoute, {
      inputText,
      parameters,
    })
      .then((value) => {
        if (cancelTokenRef.current === cancelToken) {
          const { text, metadata } = value;

          // TODO: Consider calculating actual model call latency on the backend side
          const latency = performance.now() - startTime;

          setEvaluationOutput(text);
          const metadataWithEvaluationTime = { ...metadata, latency };

          // Prefix the metadata keys with "MLFLOW_"
          const prefixedMetadata = Object.entries(metadataWithEvaluationTime).reduce(
            (acc, [metadata_key, metadata_value]) => ({
              ...acc,
              [`MLFLOW_${metadata_key}`]: metadata_value,
            }),
            {},
          );

          setEvaluationMetadata(prefixedMetadata);
          setOutputDirty(false);
          setIsEvaluating(false);
          // NB: Not using .finally() due to issues with promise implementation in the Jest
          if (cancelTokenRef.current === cancelToken) {
            cancelTokenRef.current = null;
          }
        }
      })
      .catch((e) => {
        const errorMessage = getPromptEngineeringErrorMessage(e);
        const wrappedMessage = intl.formatMessage(
          {
            defaultMessage: 'MLflow deployment returned the following error: "{errorMessage}"',
            description: 'Experiment page > MLflow deployment error message',
          },
          {
            errorMessage,
          },
        );
        // We treat is as a user error and we're not logging the error upstream
        Utils.displayGlobalErrorNotification(wrappedMessage);
        setIsEvaluating(false);
        setLastEvaluationError(wrappedMessage);
        // NB: Not using .finally() due to issues with promise implementation in the Jest
        if (cancelTokenRef.current === cancelToken) {
          cancelTokenRef.current = null;
        }
      });
  }, [inputVariableValues, modelRoutesUnified, parameters, promptTemplate, selectedModel, intl]);

  // create a handleCancel function to terminate the evaluation if it is in progress
  const handleCancel = useCallback(() => {
    if (cancelTokenRef.current) {
      setIsEvaluating(false);
      cancelTokenRef.current = null;
    }
  }, [setIsEvaluating]);

  const selectModelLabel = intl.formatMessage({
    defaultMessage: 'Served LLM model',
    description: 'Experiment page > new run modal > served LLM model endpoint label',
  });
  const selectModelPlaceholder = intl.formatMessage({
    defaultMessage: 'Select LLM model endpoint',
    description: 'Experiment page > new run modal > served LLM model endpoint placeholder',
  });

  const promptTemplateProvided = promptTemplate.trim().length > 0;
  const allInputValuesProvided = useMemo(
    () => inputVariables.every((variable) => inputVariableValues[variable]?.trim()),
    [inputVariables, inputVariableValues],
  );

  const runNameProvided = newRunName.trim().length > 0;

  // We can evaluate if we have selected model, prompt template and all input values.
  // It should be possible to evaluate without input variables for the purpose of playing around.
  const evaluateButtonEnabled = selectedModel && promptTemplateProvided && allInputValuesProvided;

  // We can log the run if we have: selected model, prompt template, all input values,
  // output that is present and up-to-date. Also, in order to log the run, we should have at least
  // one input variable defined (otherwise prompt engineering won't make sense).
  const createRunButtonEnabled = Boolean(
    selectedModel &&
      promptTemplateProvided &&
      allInputValuesProvided &&
      evaluationOutput &&
      !outputDirty &&
      inputVariables.length > 0 &&
      runNameProvided &&
      !lastEvaluationError,
  );

  // Let's prepare a proper tooltip content for every scenario
  const createRunButtonTooltip = useMemo(() => {
    if (!selectedModel) {
      return intl.formatMessage({
        defaultMessage: 'You need to select a served model endpoint using dropdown first',
        description: 'Experiment page > new run modal > invalid state - no model endpoint selected',
      });
    }
    if (!promptTemplateProvided) {
      return intl.formatMessage({
        defaultMessage: 'You need to provide a prompt template',
        description: 'Experiment page > new run modal > invalid state - no prompt template provided',
      });
    }
    if (!allInputValuesProvided) {
      return intl.formatMessage({
        defaultMessage: 'You need to provide values for all defined inputs',
        description: 'Experiment page > new run modal > invalid state - no prompt inputs provided',
      });
    }
    if (!evaluationOutput) {
      return intl.formatMessage({
        defaultMessage: 'You need to evaluate the resulting output first',
        description: 'Experiment page > new run modal > invalid state - result not evaluated',
      });
    }
    if (outputDirty) {
      return intl.formatMessage({
        defaultMessage: 'Input data or prompt template have changed since last evaluation of the output',
        description: 'Experiment page > new run modal > dirty output (out of sync with new data)',
      });
    }
    if (inputVariables.length === 0) {
      return intl.formatMessage({
        defaultMessage: 'You need to define at least one input variable',
        description: 'Experiment page > new run modal > invalid state - no input variables defined',
      });
    }
    if (!runNameProvided) {
      return intl.formatMessage({
        defaultMessage: 'Please provide run name',
        description: 'Experiment page > new run modal > invalid state - no run name provided',
      });
    }
    return null;
  }, [
    allInputValuesProvided,
    inputVariables.length,
    intl,
    outputDirty,
    evaluationOutput,
    promptTemplateProvided,
    selectedModel,
    runNameProvided,
  ]);

  // Let's prepare a proper tooltip content for every scenario
  const evaluateButtonTooltip = useMemo(() => {
    if (!selectedModel) {
      return intl.formatMessage({
        defaultMessage: 'You need to select a served model endpoint using dropdown first',
        description: 'Experiment page > new run modal > invalid state - no model endpoint selected',
      });
    }
    if (!promptTemplateProvided) {
      return intl.formatMessage({
        defaultMessage: 'You need to provide a prompt template',
        description: 'Experiment page > new run modal > invalid state - no prompt template provided',
      });
    }
    if (!allInputValuesProvided) {
      return intl.formatMessage({
        defaultMessage: 'You need to provide values for all defined inputs',
        description: 'Experiment page > new run modal > invalid state - no prompt inputs provided',
      });
    }
    return null;
  }, [allInputValuesProvided, intl, promptTemplateProvided, selectedModel]);

  if (isOpen && isViewExamplesModalOpen) {
    return (
      <EvaluationCreatePromptRunModalExamples
        isOpen={isOpen && isViewExamplesModalOpen}
        closeExamples={() => setViewExamplesModalOpen(false)}
        closeModal={closeModal}
        updatePromptTemplate={updatePromptTemplate}
        updateInputVariableValue={updateInputVariableValue}
      />
    );
  }

  const formatVisibleRouteName = (selectedRouteName: string) => {
    if (selectedRouteName.includes(':')) {
      return selectedRouteName.split(':')[1];
    }
    return selectedRouteName;
  };

  const getRouteOptionList = () => {
    return supportedModelRouteListUnified.map((modelRoute) => (
      <DialogComboboxOptionListSelectItem
        value={modelRoute.key}
        key={modelRoute.key}
        onChange={(value) => {
          updateSelectedModel(value);
        }}
        checked={selectedModel === modelRoute.key}
      >
        {modelRoute.name}
        {modelRoute.mlflowDeployment && (
          <DialogComboboxHintRow>{modelRoute.mlflowDeployment.model.name}</DialogComboboxHintRow>
        )}
      </DialogComboboxOptionListSelectItem>
    ));
  };

  return (
    <Modal
      componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_541"
      verticalSizing="maxed_out"
      visible={isOpen}
      onCancel={closeModal}
      onOk={closeModal}
      footer={
        <div css={{ display: 'flex', gap: theme.spacing.sm, justifyContent: 'flex-end' }}>
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_589"
            onClick={closeModal}
          >
            <FormattedMessage
              defaultMessage="Cancel"
              description="Experiment page > new run modal > cancel button label"
            />
          </Button>
          <LegacyTooltip title={createRunButtonTooltip}>
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_596"
              onClick={onHandleSubmit}
              data-testid="button-create-run"
              type="primary"
              disabled={!createRunButtonEnabled}
            >
              <FormattedMessage
                defaultMessage="Create run"
                description='Experiment page > new run modal > "Create run" confirm button label'
              />
            </Button>
          </LegacyTooltip>
        </div>
      }
      title={
        <div>
          <Typography.Title level={2} css={{ marginTop: theme.spacing.sm, marginBottom: theme.spacing.xs }}>
            <FormattedMessage defaultMessage="New run" description="Experiment page > new run modal > modal title" />
          </Typography.Title>
          <Typography.Hint css={{ marginTop: 0, fontWeight: 'normal' }}>
            Create a new run using a large-language model by giving it a prompt template and model parameters
          </Typography.Hint>
        </div>
      }
      dangerouslySetAntdProps={{ width: 1200 }}
    >
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: '300px 1fr',
          gap: 48,
        }}
      >
        <div>
          <FormUI.Label htmlFor="selected_model" css={{ marginBottom: theme.spacing.sm }}>
            {selectModelLabel}
          </FormUI.Label>
          <div css={{ marginBottom: theme.spacing.lg, display: 'flex', alignItems: 'center' }}>
            <DialogCombobox
              componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_597"
              label={selectModelLabel}
              modal={false}
              value={selectedModel ? [formatVisibleRouteName(selectedModel)] : undefined}
            >
              <DialogComboboxTrigger
                id="selected_model"
                css={{ width: '100%' }}
                allowClear={false}
                placeholder={selectModelPlaceholder}
                withInlineLabel={false}
              />
              <DialogComboboxContent loading={modelRoutesLoading} maxHeight={400} matchTriggerWidth>
                {!modelRoutesLoading && (
                  <DialogComboboxOptionList>
                    <DialogComboboxOptionListSearch autoFocus>{getRouteOptionList()}</DialogComboboxOptionListSearch>
                  </DialogComboboxOptionList>
                )}
              </DialogComboboxContent>
            </DialogCombobox>
          </div>
          {selectedModel && (
            <EvaluationCreatePromptParameters parameters={parameters} updateParameter={updateParameter} />
          )}
          <div css={styles.formItem}>
            <>
              <FormUI.Label htmlFor="new_run_name">
                <FormattedMessage
                  defaultMessage="New run name"
                  description="Experiment page > new run modal > run name input label"
                />
                {!newRunName.trim() && (
                  <FormUI.Message
                    type="error"
                    message={intl.formatMessage({
                      defaultMessage: 'Please provide run name',
                      description: 'Experiment page > new run modal > invalid state - no run name provided',
                    })}
                  />
                )}
              </FormUI.Label>
              <Input
                componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_638"
                id="new_run_name"
                data-testid="run-name-input"
                required
                value={newRunName}
                onChange={(e) => setNewRunName(e.target.value)}
              />
            </>
          </div>
        </div>
        <div>
          <div css={styles.formItem}>
            <>
              <div css={{ display: 'flex', justifyContent: 'space-between' }}>
                <FormUI.Label htmlFor="prompt_template">
                  <FormattedMessage
                    defaultMessage="Prompt Template"
                    description="Experiment page > new run modal > prompt template input label"
                  />
                </FormUI.Label>
                <Button
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_695"
                  onClick={() => setViewExamplesModalOpen(true)}
                  style={{ marginLeft: 'auto' }}
                  size="small"
                >
                  <FormattedMessage
                    defaultMessage="View Examples"
                    description="Experiment page > new run modal > prompt examples button"
                  />
                </Button>
              </div>
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage={`Give instructions to the model. Use '{{ }}' or the "Add new variable" button to add variables to your prompt.`}
                  description="Experiment page > new run modal > prompt template input hint"
                />
              </FormUI.Hint>
            </>

            <TextArea
              componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_678"
              id="prompt_template"
              autoSize={{ minRows: 3 }}
              data-testid="prompt-template-input"
              value={promptTemplate}
              onChange={(e) => updatePromptTemplate(e.target.value)}
              ref={savePromptTemplateInputRef}
            />
            <EvaluationCreateRunPromptTemplateErrors violations={inputVariableNameViolations} />
          </div>
          {inputVariables.map((inputVariable) => (
            <div css={styles.formItem} key={inputVariable}>
              <>
                <FormUI.Label htmlFor={inputVariable}>
                  <span>{inputVariable}</span>
                </FormUI.Label>
                <TextArea
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_694"
                  id={inputVariable}
                  autoSize
                  value={inputVariableValues[inputVariable] ? inputVariableValues[inputVariable] : ''}
                  onChange={(e) => updateInputVariableValue(inputVariable, e.target.value)}
                />
              </>
            </div>
          ))}
          <div css={{ marginBottom: 2 * theme.spacing.md }}>
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodal.tsx_736"
              icon={<PlusIcon />}
              onClick={handleAddVariableToTemplate}
            >
              <FormattedMessage
                defaultMessage="Add new variable"
                description='Experiment page > new run modal > "add new variable" button label'
              />
            </Button>
          </div>
          <EvaluationCreatePromptRunOutput
            evaluateButtonTooltip={evaluateButtonTooltip}
            evaluationMetadata={evaluationMetadata}
            evaluationOutput={evaluationOutput}
            disabled={!evaluateButtonEnabled}
            isEvaluating={isEvaluating}
            isOutputDirty={outputDirty}
            onCancelClick={handleCancel}
            onEvaluateClick={handleEvaluate}
            evaluationError={lastEvaluationError}
          />
        </div>
      </div>
      {isCreatingRun && (
        // Scrim overlay
        <div
          css={{
            inset: 0,
            backgroundColor: theme.colors.overlayOverlay,
            position: 'absolute',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1,
          }}
        >
          <Spinner />
        </div>
      )}
    </Modal>
  );
};

const styles = {
  formItem: { marginBottom: 16 },
};

import {
  Alert,
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxHintRow,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  InfoIcon,
  Input,
  Modal,
  PlayIcon,
  PlusIcon,
  Spinner,
  StopIcon,
  TableSkeleton,
  Tooltip,
  Typography,
  WarningIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { sortBy } from 'lodash';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDispatch, useSelector } from 'react-redux';
import Utils from '../../../common/utils/Utils';
import { ThunkDispatch } from '../../../redux-types';
import { createPromptLabRunApi } from '../../actions';
import {
  queryModelGatewayRouteApi,
  searchModelGatewayRoutesApi,
} from '../../actions/ModelGatewayActions';
import { ModelGatewayReduxState } from '../../reducers/ModelGatewayReducer';
import {
  GatewayErrorWrapper,
  ModelGatewayResponseType,
  ModelGatewayRouteType,
  ModelGatewayService,
} from '../../sdk/ModelGatewayService';
import { generateRandomRunName, getDuplicatedRunName } from '../../utils/RunNameUtils';
import { useExperimentIds } from '../experiment-page/hooks/useExperimentIds';
import { useFetchExperimentRuns } from '../experiment-page/hooks/useFetchExperimentRuns';
import {
  PROMPTLAB_METADATA_COLUMN_LATENCY,
  PROMPTLAB_METADATA_COLUMN_TOTAL_TOKENS,
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

const { TextArea } = Input;
type Props = {
  isOpen: boolean;
  closeModal: () => void;
  runBeingDuplicated: RunRowType | null;
  visibleRuns?: RunRowType[];
};

export const EvaluationCreatePromptRunModal = ({
  isOpen,
  closeModal,
  runBeingDuplicated,
  visibleRuns = [],
}: Props): JSX.Element => {
  const [experimentId] = useExperimentIds();
  const { theme } = useDesignSystemTheme();
  const { parameters, updateParameter } = usePromptEvaluationParameters();

  const [selectedModel, updateSelectedModel] = useState('');
  const [newRunName, setNewRunName] = useState('');
  const [isCreatingRun, setIsCreatingRun] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [lastEvaluationError, setLastEvaluationError] = useState<string | null>(null);
  const [evaluationOutput, setEvaluationOutput] = useState('');
  const [evaluationMetadata, setEvaluationMetadata] = useState<
    Partial<ModelGatewayResponseType['metadata']>
  >({});
  const [outputDirty, setOutputDirty] = useState(false);
  const [isViewExamplesModalOpen, setViewExamplesModalOpen] = useState(false);
  const cancelTokenRef = useRef<string | null>(null);

  const dispatch = useDispatch<ThunkDispatch>();

  useEffect(() => {
    dispatch(searchModelGatewayRoutesApi()).catch((e) => {
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

  const {
    handleAddVariableToTemplate,
    savePromptTemplateInputRef,
    promptTemplate,
    updatePromptTemplate,
  } = usePromptEvaluationPromptTemplateValue();

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
        visibleRuns.map(({ runName }) => runName),
      );
      setNewRunName(duplicatedRunName);
      clearInputVariableValues();
    }
  }, [
    runBeingDuplicated,
    clearInputVariableValues,
    updateParameter,
    updatePromptTemplate,
    visibleRuns,
  ]);

  // Select model gateway routes from the state
  const modelRoutes = useSelector(
    ({ modelGateway }: { modelGateway: ModelGatewayReduxState }) => modelGateway.modelGatewayRoutes,
  );

  // Limit model routes to "COMPLETIONS" and "CHAT" types only and sort them alphabetically
  const supportedModelRouteList = useMemo(
    () =>
      sortBy(
        Object.values(modelRoutes).filter((modelRoute) =>
          [ModelGatewayRouteType.LLM_V1_COMPLETIONS, ModelGatewayRouteType.LLM_V1_CHAT].includes(
            modelRoute.route_type,
          ),
        ),
        'name',
      ),
    [modelRoutes],
  );

  // Determines if model gateway routes are being loaded
  const modelRoutesLoading = useSelector(
    ({ modelGateway }: { modelGateway: ModelGatewayReduxState }) =>
      modelGateway.modelGatewayRoutesLoading,
  );
  useEffect(() => {
    if (evaluationOutput) {
      setOutputDirty(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputVariableValues, promptTemplate, parameters, selectedModel]);

  const { refreshRuns, updateSearchFacets } = useFetchExperimentRuns();

  const onHandleSubmit = () => {
    setIsCreatingRun(true);
    const modelInput = compilePromptInputText(promptTemplate, inputVariableValues);
    dispatch(
      createPromptLabRunApi({
        experimentId,
        promptTemplate,
        modelInput,
        modelParameters: parameters,
        modelRouteName: selectedModel,
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

        // If the view if not in the "evaluation" mode already, open it
        updateSearchFacets((currentState) => {
          if (currentState.compareRunsMode !== 'ARTIFACT') {
            return { ...currentState, compareRunsMode: 'ARTIFACT' };
          }
          return currentState;
        });
      })
      .catch((e) => {
        Utils.logErrorAndNotifyUser(e?.message || e);
        // NB: Not using .finally() due to issues with promise implementation in the Jest
        setIsCreatingRun(false);
      });
  };

  const handleEvaluate = useCallback(() => {
    const modelRoute = modelRoutes[selectedModel];
    const cancelToken = Math.random().toString(36);
    cancelTokenRef.current = cancelToken;
    if (!modelRoute) {
      // Should never happen if the model is selected
      throw new Error('No model route found!');
    }
    setLastEvaluationError(null);
    setIsEvaluating(true);
    const inputText = compilePromptInputText(promptTemplate, inputVariableValues);
    dispatch(
      queryModelGatewayRouteApi(modelRoute, {
        inputText,
        parameters,
      }),
    )
      .then(({ value, action }) => {
        if (cancelTokenRef.current === cancelToken) {
          const { text, metadata } = ModelGatewayService.parseEvaluationResponse(value);

          // TODO: Consider calculating actual model call latency on the backend side
          const latency = performance.now() - action.meta.startTime;

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
      .catch((e: GatewayErrorWrapper) => {
        const errorMessage = e.getGatewayErrorMessage() || e.getUserVisibleError();
        const wrappedMessage = intl.formatMessage(
          {
            defaultMessage: 'AI gateway returned the following error: "{errorMessage}"',
            description: 'Experiment page > new run modal > AI gateway error message',
          },
          {
            errorMessage,
          },
        );
        Utils.logErrorAndNotifyUser(wrappedMessage);
        setIsEvaluating(false);
        setLastEvaluationError(wrappedMessage);
        // NB: Not using .finally() due to issues with promise implementation in the Jest
        if (cancelTokenRef.current === cancelToken) {
          cancelTokenRef.current = null;
        }
      });
  }, [dispatch, inputVariableValues, modelRoutes, parameters, promptTemplate, selectedModel, intl]);

  // create a handleCancel function to terminate the evaluation if it is in progress
  const handleCancel = useCallback(() => {
    if (cancelTokenRef.current) {
      setIsEvaluating(false);
      cancelTokenRef.current = null;
    }
  }, [setIsEvaluating]);

  const selectModelLabel = intl.formatMessage({
    defaultMessage: 'Model',
    description: 'Experiment page > new run modal > AI gateway selector label',
  });
  const selectModelPlaceholder = intl.formatMessage({
    defaultMessage: 'Select route',
    description: 'Experiment page > new run modal > AI gateway selector placeholder',
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
        defaultMessage: 'You need to select a route from the AI gateway dropdown first',
        description: 'Experiment page > new run modal > invalid state - no AI gateway selected',
      });
    }
    if (!promptTemplateProvided) {
      return intl.formatMessage({
        defaultMessage: 'You need to provide a prompt template',
        description:
          'Experiment page > new run modal > invalid state - no prompt template provided',
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
        defaultMessage:
          'Input data or prompt template have changed since last evaluation of the output',
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
        defaultMessage: 'You need to select a route from the AI gateway dropdown first',
        description: 'Experiment page > new run modal > invalid state - no AI gateway selected',
      });
    }
    if (!promptTemplateProvided) {
      return intl.formatMessage({
        defaultMessage: 'You need to provide a prompt template',
        description:
          'Experiment page > new run modal > invalid state - no prompt template provided',
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

  const evaluateOutput = useMemo(() => {
    if (lastEvaluationError) {
      return (
        <Alert
          message={lastEvaluationError}
          closable={false}
          type='error'
          css={{ marginBottom: theme.spacing.sm, marginTop: theme.spacing.sm }}
        />
      );
    }
    if (isEvaluating) {
      return (
        <div css={{ marginTop: theme.spacing.sm }}>
          <TableSkeleton lines={5} />
        </div>
      );
    }
    return (
      <TextArea
        rows={5}
        css={{ cursor: 'default' }}
        data-testid='prompt-output'
        value={evaluationOutput}
        readOnly
      />
    );
  }, [lastEvaluationError, isEvaluating, evaluationOutput, theme]);

  const metadataOutput = useMemo(() => {
    if (!evaluationMetadata) {
      return null;
    }
    if (isEvaluating) {
      return null;
    }
    return (
      <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
        {PROMPTLAB_METADATA_COLUMN_LATENCY in evaluationMetadata && (
          <Typography.Hint size='sm'>
            {Math.round(Number(evaluationMetadata[PROMPTLAB_METADATA_COLUMN_LATENCY]))} ms
            {'MLFLOW_total_tokens' in evaluationMetadata ? ',' : ''}
          </Typography.Hint>
        )}
        {PROMPTLAB_METADATA_COLUMN_TOTAL_TOKENS in evaluationMetadata && (
          <Typography.Hint size='sm'>
            <FormattedMessage
              defaultMessage='{totalTokens} total tokens'
              description='Experiment page > artifact compare view > results table > total number of evaluated tokens'
              values={{ totalTokens: evaluationMetadata[PROMPTLAB_METADATA_COLUMN_TOTAL_TOKENS] }}
            />
          </Typography.Hint>
        )}
      </div>
    );
  }, [evaluationMetadata, evaluationOutput, isEvaluating, theme]);

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

  return (
    <Modal
      verticalSizing='maxed_out'
      visible={isOpen}
      onCancel={closeModal}
      onOk={closeModal}
      footer={
        <div css={{ display: 'flex', gap: theme.spacing.sm, justifyContent: 'flex-end' }}>
          <Button onClick={closeModal}>
            <FormattedMessage
              defaultMessage='Cancel'
              description='Experiment page > new run modal > cancel button label'
            />
          </Button>
          <Tooltip title={createRunButtonTooltip}>
            <Button
              onClick={onHandleSubmit}
              data-testid='button-create-run'
              type='primary'
              disabled={!createRunButtonEnabled}
            >
              <FormattedMessage
                defaultMessage='Create run'
                description='Experiment page > new run modal > "Create run" confirm button label'
              />
            </Button>
          </Tooltip>
        </div>
      }
      title={
        <div>
          <Typography.Title
            level={2}
            css={{ marginTop: theme.spacing.sm, marginBottom: theme.spacing.xs }}
          >
            <FormattedMessage
              defaultMessage='New run'
              description='Experiment page > new run modal > modal title'
            />
          </Typography.Title>
          <Typography.Hint css={{ marginTop: 0, fontWeight: 'normal' }}>
            Create a new run using a large-language model by giving it a prompt template and model
            parameters
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
          <div css={{ ...styles.formItem, display: 'flex', alignItems: 'center' }}>
            <DialogCombobox
              label={selectedModel ? selectModelLabel : selectModelPlaceholder}
              modal={false}
              value={selectedModel ? [selectedModel] : undefined}
            >
              <DialogComboboxTrigger css={{ width: '100%' }} allowClear={false} />
              <DialogComboboxContent loading={modelRoutesLoading} maxHeight={400} matchTriggerWidth>
                {!modelRoutesLoading && (
                  <DialogComboboxOptionList>
                    <DialogComboboxOptionListSearch autoFocus>
                      {supportedModelRouteList.map((modelRoute) => (
                        <DialogComboboxOptionListSelectItem
                          value={modelRoute.name}
                          key={modelRoute.name}
                          onChange={(value) => {
                            updateSelectedModel(value);
                          }}
                          checked={selectedModel === modelRoute.name}
                        >
                          {modelRoute.name}
                          <DialogComboboxHintRow>{modelRoute.model.name}</DialogComboboxHintRow>
                        </DialogComboboxOptionListSelectItem>
                      ))}
                    </DialogComboboxOptionListSearch>
                  </DialogComboboxOptionList>
                )}
              </DialogComboboxContent>
            </DialogCombobox>
            <Tooltip
              title={
                <FormattedMessage
                  defaultMessage={
                    'These routes come from the AI Gateway. Check out the AI Gateway preview documentation to get started'
                  }
                  description={'Information about gateway routes'}
                />
              }
              placement='right'
            >
              <InfoIcon
                css={{
                  marginLeft: theme.spacing.sm,
                  color: theme.colors.textSecondary,
                }}
              />
            </Tooltip>
          </div>
          {selectedModel && (
            <EvaluationCreatePromptParameters
              parameters={parameters}
              updateParameter={updateParameter}
            />
          )}
          <div css={styles.formItem}>
            <>
              <FormUI.Label htmlFor={'new_run_name'}>
                <FormattedMessage
                  defaultMessage='New run name'
                  description='Experiment page > new run modal > run name input label'
                />
                {!newRunName.trim() && (
                  <FormUI.Message
                    type='error'
                    message={intl.formatMessage({
                      defaultMessage: 'Please provide run name',
                      description:
                        'Experiment page > new run modal > invalid state - no run name provided',
                    })}
                  />
                )}
              </FormUI.Label>
              <Input
                id='new_run_name'
                data-testid='run-name-input'
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
              <FormUI.Label htmlFor={'prompt_template'}>
                <FormattedMessage
                  defaultMessage='Prompt Template'
                  description='Experiment page > new run modal > prompt template input label'
                />
                <Button
                  onClick={() => setViewExamplesModalOpen(true)}
                  style={{ marginLeft: 'auto' }}
                >
                  <FormattedMessage
                    defaultMessage='View Examples'
                    description='Experiment page > new run modal > prompt examples button'
                  />
                </Button>
              </FormUI.Label>
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage={`Give instructions to the model. Use '{{ }}' or the "Add new variable" button to add variables to your prompt.`}
                  description='Experiment page > new run modal > prompt template input hint'
                />
              </FormUI.Hint>
            </>

            <TextArea
              id='prompt_template'
              autoSize
              data-testid='prompt-template-input'
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
                  id={inputVariable}
                  autoSize
                  value={
                    inputVariableValues[inputVariable] ? inputVariableValues[inputVariable] : ''
                  }
                  onChange={(e) => updateInputVariableValue(inputVariable, e.target.value)}
                />
              </>
            </div>
          ))}
          <div css={styles.formItem}>
            <Button icon={<PlusIcon />} onClick={handleAddVariableToTemplate}>
              <FormattedMessage
                defaultMessage='Add new variable'
                description='Experiment page > new run modal > "add new variable" button label'
              />
            </Button>
          </div>
          <div css={styles.formItem}>
            <Tooltip title={evaluateButtonTooltip}>
              <Button
                data-testid='button-evaluate'
                icon={<PlayIcon />}
                onClick={handleEvaluate}
                disabled={!evaluateButtonEnabled}
                loading={isEvaluating}
              >
                <FormattedMessage
                  defaultMessage='Evaluate'
                  description='Experiment page > new run modal > "evaluate" button label'
                />
              </Button>
            </Tooltip>
            <Button
              data-testid='button-cancel'
              icon={<StopIcon />}
              onClick={handleCancel}
              disabled={!isEvaluating}
              css={{ marginLeft: theme.spacing.sm }}
            >
              <FormattedMessage
                defaultMessage='Cancel'
                description='Experiment page > new run modal > "cancel" button label'
              />
            </Button>
          </div>
          <FormUI.Label>
            <FormattedMessage
              defaultMessage='Output'
              description='Experiment page > new run modal > evaluation output field label'
            />{' '}
            {outputDirty && (
              <Tooltip
                title={
                  <FormattedMessage
                    defaultMessage='Model, input data or prompt have changed since last evaluation of the output'
                    description='Experiment page > new run modal > dirty output (out of sync with new data)'
                  />
                }
              >
                <WarningIcon />
              </Tooltip>
            )}
          </FormUI.Label>
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage='This is the output generated by the LLM using the prompt template and input values defined above.'
              description='Experiment page > new run modal > evaluation output field hint'
            />
          </FormUI.Hint>
          {evaluateOutput}
          {metadataOutput}
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

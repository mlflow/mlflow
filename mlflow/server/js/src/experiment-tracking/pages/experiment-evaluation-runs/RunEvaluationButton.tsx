import {
  Alert,
  Button,
  ChartLineIcon,
  Checkbox,
  Input,
  Modal,
  PillControl,
  SearchIcon,
  TableSkeleton,
  Typography,
  getShadowScrollStyles,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { createTraceLocationForExperiment, useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';
import { isEmpty } from 'lodash';
import { useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { ExperimentPageTabName } from '../../constants';
import { SELECTED_RUN_UUID_QUERY_PARAM } from '../../components/evaluations/hooks/useSelectedRunUuid';
import { EndpointSelector } from '../../components/EndpointSelector';
import { SelectTracesModal } from '../../components/SelectTracesModal';
import { useMonitoringFiltersTimeRange } from '../../hooks/useMonitoringFilters';
import { formatGatewayModelFromEndpoint, getEndpointNameFromGatewayModel } from '../../../gateway/utils/gatewayUtils';
import { ScorerEvaluationScope } from '../experiment-scorers/constants';
import { useGetScheduledScorers } from '../experiment-scorers/hooks/useGetScheduledScorers';
import { useTemplateOptions } from '../experiment-scorers/llmScorerUtils';
import { TEMPLATE_INSTRUCTIONS_MAP } from '../experiment-scorers/prompts';
import { LLM_TEMPLATE, type LLMScorer, type ScheduledScorer } from '../experiment-scorers/types';
import { transformScheduledScorer } from '../experiment-scorers/utils/scorerTransformUtils';
import { useInvokeGenAIEvaluation } from './hooks/useInvokeGenAIEvaluation';

type JudgeSelectionMode = 'llm' | 'template';

const isTraceLevelLLMScorer = (scorer: ScheduledScorer): scorer is LLMScorer =>
  scorer.type === 'llm' && !scorer.isSessionLevelScorer;

export const RunEvaluationButton = ({ experimentId }: { experimentId: string }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { displayMap, templateOptions } = useTemplateOptions(ScorerEvaluationScope.TRACES);
  const [isOpen, setIsOpen] = useState(false);
  const [selectedTraceIds, setSelectedTraceIds] = useState<string[]>([]);
  const [isSelectTracesModalOpen, setIsSelectTracesModalOpen] = useState(false);
  const [selectedScorers, setSelectedScorers] = useState<LLMScorer[]>([]);
  const [selectedTemplates, setSelectedTemplates] = useState<LLM_TEMPLATE[]>([]);
  const [currentEndpointModel, setCurrentEndpointModel] = useState<string | undefined>(undefined);
  const hasSelectedTemplates = selectedTemplates.length > 0;
  const selectedJudgeCount = selectedScorers.length + selectedTemplates.length;
  const runJudgeDisabled =
    selectedTraceIds.length === 0 || selectedJudgeCount === 0 || (hasSelectedTemplates && !currentEndpointModel);

  const {
    mutate: invokeGenAIEvaluation,
    isLoading: isSubmitting,
    error: submitError,
    reset: resetSubmit,
  } = useInvokeGenAIEvaluation();

  const [buildError, setBuildError] = useState<Error | null>(null);
  const submissionError = submitError ?? buildError;

  const clearSubmissionError = () => {
    resetSubmit();
    setBuildError(null);
  };

  const toggleScorer = (scorer: LLMScorer) => {
    setSelectedScorers((prev) => {
      const isSelected = prev.some((s) => s.name === scorer.name);
      return isSelected ? prev.filter((s) => s.name !== scorer.name) : [...prev, scorer];
    });
  };

  const toggleTemplate = (template: LLM_TEMPLATE) => {
    setSelectedTemplates((prev) => {
      const isSelected = prev.includes(template);
      return isSelected ? prev.filter((t) => t !== template) : [...prev, template];
    });
  };

  const traceSearchLocations = useMemo(() => [createTraceLocationForExperiment(experimentId)], [experimentId]);
  const timeRange = useMonitoringFiltersTimeRange();
  const { data: traceInfos, isLoading: isLoadingTraces } = useSearchMlflowTraces({
    locations: traceSearchLocations,
    timeRange,
    disabled: !isOpen,
  });

  const hasSeededSelectionRef = useRef(false);
  useEffect(() => {
    if (!isOpen) {
      hasSeededSelectionRef.current = false;
      return;
    }
    if (hasSeededSelectionRef.current || isLoadingTraces || !traceInfos) {
      return;
    }
    hasSeededSelectionRef.current = true;
    setSelectedTraceIds(traceInfos.map((trace) => trace.trace_id));
  }, [isOpen, isLoadingTraces, traceInfos]);

  const closeAndResetSelections = () => {
    setIsOpen(false);
    setSelectedScorers([]);
    setSelectedTemplates([]);
    setCurrentEndpointModel(undefined);
    setBuildError(null);
    resetSubmit();
  };

  const buildSerializedScorers = (): string[] => {
    const fromCustom = selectedScorers.map((scorer) => transformScheduledScorer(scorer).serialized_scorer);

    const fromTemplates = selectedTemplates.map((template) => {
      const instructions = TEMPLATE_INSTRUCTIONS_MAP[template];
      const adHocScheduledScorer: ScheduledScorer = instructions
        ? {
            name: displayMap[template] ?? template,
            type: 'llm',
            llmTemplate: LLM_TEMPLATE.CUSTOM,
            instructions,
            model: currentEndpointModel,
            is_instructions_judge: true,
            isSessionLevelScorer: false,
          }
        : {
            name: displayMap[template] ?? template,
            type: 'llm',
            llmTemplate: template,
            model: currentEndpointModel,
            is_instructions_judge: false,
            isSessionLevelScorer: false,
          };
      return transformScheduledScorer(adHocScheduledScorer).serialized_scorer;
    });

    return [...fromCustom, ...fromTemplates];
  };

  const handleSubmit = () => {
    let serializedScorers: string[];
    try {
      serializedScorers = buildSerializedScorers();
    } catch (error) {
      setBuildError(error instanceof Error ? error : new Error(String(error)));
      return;
    }
    setBuildError(null);

    invokeGenAIEvaluation(
      {
        experimentId,
        traceIds: selectedTraceIds,
        serializedScorers,
      },
      {
        onSuccess: (response) => {
          queryClient.invalidateQueries({ queryKey: ['SEARCH_RUNS', experimentId] });
          closeAndResetSelections();
          navigate({
            pathname: Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns),
            search: `?${SELECTED_RUN_UUID_QUERY_PARAM}=${response.run_id}`,
          });
        },
      },
    );
  };

  return (
    <>
      <Button
        componentId="mlflow.eval-runs.start-run-button"
        icon={<ChartLineIcon />}
        type="primary"
        onClick={() => setIsOpen(true)}
      >
        <FormattedMessage
          defaultMessage="Evaluate traces"
          description="Label for the primary button that opens the run-evaluation modal"
        />
      </Button>
      <Modal
        componentId="mlflow.eval-runs.start-run-modal"
        title={
          <FormattedMessage defaultMessage="Run evaluation" description="Title for the run evaluation modal dialog" />
        }
        visible={isOpen}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel button text in the run evaluation modal',
        })}
        okText={
          selectedJudgeCount > 1
            ? intl.formatMessage({
                defaultMessage: 'Run judges',
                description: 'Button text for running multiple judges from the run evaluation modal',
              })
            : intl.formatMessage({
                defaultMessage: 'Run judge',
                description: 'Button text for running a single judge from the run evaluation modal',
              })
        }
        okButtonProps={{ disabled: runJudgeDisabled, loading: isSubmitting }}
        cancelButtonProps={{ disabled: isSubmitting }}
        onOk={handleSubmit}
        onCancel={isSubmitting ? undefined : closeAndResetSelections}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {submissionError && (
            <Alert
              componentId="mlflow.eval-runs.start-run-modal.error"
              type="error"
              message={submissionError.message}
              closable
              onClose={clearSubmissionError}
            />
          )}
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Traces"
                description="Section header for trace selection in the run evaluation modal"
              />
            </Typography.Text>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Select the traces to evaluate."
                description="Description for the trace selection section in the run evaluation modal"
              />
            </Typography.Text>
            <div>
              <Button
                componentId="mlflow.eval-runs.start-run-modal.select-traces"
                onClick={() => setIsSelectTracesModalOpen(true)}
                loading={isLoadingTraces && !hasSeededSelectionRef.current}
              >
                {selectedTraceIds.length > 0 ? (
                  <FormattedMessage
                    defaultMessage="{count, plural, one {1 trace selected} other {# traces selected}}"
                    description="Label showing number of traces selected in the run evaluation modal"
                    values={{ count: selectedTraceIds.length }}
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Select traces"
                    description="Button to open the trace selection modal from the run evaluation modal"
                  />
                )}
              </Button>
            </div>
          </div>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Judges"
                description="Section header for judge selection in the run evaluation modal"
              />
            </Typography.Text>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Select the LLM-as-a-judge scorers to run on the selected traces."
                description="Description for the judge selection section in the run evaluation modal"
              />
            </Typography.Text>
            <JudgesSelectionSection
              experimentId={experimentId}
              enabled={isOpen}
              templateOptions={templateOptions}
              selectedScorers={selectedScorers}
              selectedTemplates={selectedTemplates}
              onToggleScorer={toggleScorer}
              onToggleTemplate={toggleTemplate}
            />
            {hasSelectedTemplates && (
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text bold>
                  <FormattedMessage
                    defaultMessage="Endpoint"
                    description="Label for the endpoint-selection section in the run evaluation modal"
                  />
                </Typography.Text>
                <Typography.Text color="secondary">
                  <FormattedMessage
                    defaultMessage="Select the model endpoint that the pre-built LLM-as-a-judge scorers will run against."
                    description="Description for the endpoint-selection section in the run evaluation modal"
                  />
                </Typography.Text>
                <EndpointSelector
                  currentEndpointName={getEndpointNameFromGatewayModel(currentEndpointModel)}
                  onEndpointSelect={(endpointName) => {
                    setCurrentEndpointModel(formatGatewayModelFromEndpoint(endpointName));
                  }}
                  autoSelectFirstEndpoint
                />
              </div>
            )}
          </div>
        </div>
      </Modal>
      {isSelectTracesModalOpen && (
        <SelectTracesModal
          onClose={() => setIsSelectTracesModalOpen(false)}
          onSuccess={(traceIds) => {
            setSelectedTraceIds(traceIds);
            setIsSelectTracesModalOpen(false);
          }}
          initialTraceIdsSelected={selectedTraceIds}
        />
      )}
    </>
  );
};

const HIDDEN_PRE_BUILT_TEMPLATES = [LLM_TEMPLATE.CUSTOM, LLM_TEMPLATE.GUIDELINES];

const JudgesSelectionSection = ({
  experimentId,
  enabled,
  templateOptions,
  selectedScorers,
  selectedTemplates,
  onToggleScorer,
  onToggleTemplate,
}: {
  experimentId: string;
  enabled: boolean;
  templateOptions: ReturnType<typeof useTemplateOptions>['templateOptions'];
  selectedScorers: LLMScorer[];
  selectedTemplates: LLM_TEMPLATE[];
  onToggleScorer: (scorer: LLMScorer) => void;
  onToggleTemplate: (template: LLM_TEMPLATE) => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [searchValue, setSearchValue] = useState<string>('');
  const [judgeSelectionMode, setJudgeSelectionMode] = useState<JudgeSelectionMode>('llm');

  const { data, isLoading: loadingScorers } = useGetScheduledScorers(experimentId, { enabled });

  const displayedLLMScorers = useMemo(() => {
    const lowercased = searchValue.toLowerCase();
    return (data?.scheduledScorers ?? [])
      .filter(isTraceLevelLLMScorer)
      .filter((scorer) => scorer.name.toLowerCase().includes(lowercased));
  }, [data?.scheduledScorers, searchValue]);

  const displayedTemplates = useMemo(() => {
    const lowercased = searchValue.toLowerCase();
    return templateOptions.filter(
      (template) =>
        !HIDDEN_PRE_BUILT_TEMPLATES.includes(template.value) && template.label.toLowerCase().includes(lowercased),
    );
  }, [templateOptions, searchValue]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <Input
        componentId="mlflow.eval-runs.start-run-modal.judge-search"
        prefix={<SearchIcon />}
        value={searchValue}
        onChange={(e) => setSearchValue(e.target.value)}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search judges',
          description: 'Placeholder for the judge search input in the run evaluation modal',
        })}
      />
      <div>
        <PillControl.Root
          size="small"
          componentId="mlflow.eval-runs.start-run-modal.judge-type-filter"
          value={judgeSelectionMode}
          onValueChange={(value) => setJudgeSelectionMode(value as JudgeSelectionMode)}
        >
          <PillControl.Item value="llm">
            <FormattedMessage
              defaultMessage="Custom LLM-as-a-judge ({llmCount})"
              description="Label for the custom LLM-as-a-judge filter pill in the run evaluation modal"
              values={{ llmCount: displayedLLMScorers.length }}
            />
          </PillControl.Item>
          <PillControl.Item value="template">
            <FormattedMessage
              defaultMessage="Pre-built LLM-as-a-judge ({templateCount})"
              description="Label for the pre-built LLM-as-a-judge filter pill in the run evaluation modal"
              values={{ templateCount: displayedTemplates.length }}
            />
          </PillControl.Item>
        </PillControl.Root>
      </div>
      <div
        css={{
          height: 200,
          display: 'flex',
          flexDirection: 'column',
          overflowY: 'auto',
          ...getShadowScrollStyles(theme, { orientation: 'vertical' }),
        }}
      >
        {judgeSelectionMode === 'llm' &&
          (loadingScorers ? (
            <TableSkeleton lines={3} />
          ) : isEmpty(displayedLLMScorers) ? (
            <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.sm }}>
              <Typography.Hint>
                <FormattedMessage
                  defaultMessage="No custom LLM-as-a-judge scorers found"
                  description="Empty-state label when the experiment has no custom LLM-as-a-judge scorers"
                />
              </Typography.Hint>
            </div>
          ) : (
            displayedLLMScorers.map((scorer) => (
              <JudgeScorerOption
                key={scorer.name}
                scorer={scorer}
                selected={selectedScorers.some((s) => s.name === scorer.name)}
                onToggle={() => onToggleScorer(scorer)}
              />
            ))
          ))}
        {judgeSelectionMode === 'template' &&
          (isEmpty(displayedTemplates) ? (
            <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.sm }}>
              <Typography.Hint>
                <FormattedMessage
                  defaultMessage="No pre-built LLM-as-a-judge scorers found"
                  description="Empty-state label when no pre-built LLM-as-a-judge templates match the search"
                />
              </Typography.Hint>
            </div>
          ) : (
            displayedTemplates.map((template) => (
              <JudgeTemplateOption
                key={template.value}
                template={template}
                selected={selectedTemplates.includes(template.value)}
                onToggle={() => onToggleTemplate(template.value)}
              />
            ))
          ))}
      </div>
    </div>
  );
};

const judgeRowWrapperStyle = {
  width: '100%',
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  cursor: 'pointer',
} as const;

const JudgeScorerOption = ({
  scorer,
  selected,
  onToggle,
}: {
  scorer: LLMScorer;
  selected: boolean;
  onToggle: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ height: 48, flexShrink: 0, display: 'flex' }}>
      <Checkbox
        componentId="mlflow.eval-runs.start-run-modal.judge-llm"
        isChecked={selected}
        onChange={onToggle}
        wrapperStyle={judgeRowWrapperStyle}
      >
        <div css={{ display: 'flex', flexDirection: 'column', marginLeft: theme.spacing.xs }}>
          <Typography.Text css={{ flex: 1 }}>{scorer.name}</Typography.Text>
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="Custom judge"
              description="Sub-label for a custom LLM-as-a-judge row in the run evaluation modal"
            />
          </Typography.Hint>
        </div>
      </Checkbox>
    </div>
  );
};

const JudgeTemplateOption = ({
  template,
  selected,
  onToggle,
}: {
  template: { value: LLM_TEMPLATE; label: string; hint: string };
  selected: boolean;
  onToggle: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ height: 48, flexShrink: 0, display: 'flex' }}>
      <Checkbox
        componentId="mlflow.eval-runs.start-run-modal.judge-template"
        isChecked={selected}
        onChange={onToggle}
        wrapperStyle={judgeRowWrapperStyle}
      >
        <div css={{ display: 'flex', flexDirection: 'column', marginLeft: theme.spacing.xs }}>
          <Typography.Text css={{ flex: 1 }}>{template.label}</Typography.Text>
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="Pre-built LLM-as-a-judge"
              description="Sub-label for a pre-built LLM-as-a-judge row in the run evaluation modal"
            />
          </Typography.Hint>
        </div>
      </Checkbox>
    </div>
  );
};

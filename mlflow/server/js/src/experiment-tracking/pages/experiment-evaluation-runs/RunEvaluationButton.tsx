import {
  Button,
  ChartLineIcon,
  Checkbox,
  Input,
  Modal,
  PillControl,
  SearchIcon,
  Spinner,
  TableSkeleton,
  Tabs,
  Typography,
  getShadowScrollStyles,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { CodeSnippet, SnippetCopyAction } from '@mlflow/mlflow/src/shared/web-shared/snippet';
import { AggregationType, MetricViewType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { createTraceLocationForExperiment, useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';
import { isEmpty } from 'lodash';
import { useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { EndpointSelector } from '../../components/EndpointSelector';
import { SelectTracesModal } from '../../components/SelectTracesModal';
import { formatGatewayModelFromEndpoint, getEndpointNameFromGatewayModel } from '../../../gateway/utils/gatewayUtils';
import { useTraceMetricsQuery } from '../experiment-overview/hooks/useTraceMetricsQuery';
import { ScorerEvaluationScope } from '../experiment-scorers/constants';
import { useGetScheduledScorers } from '../experiment-scorers/hooks/useGetScheduledScorers';
import { useTemplateOptions } from '../experiment-scorers/llmScorerUtils';
import { LLM_TEMPLATE, type LLMScorer } from '../experiment-scorers/types';

const getDatasetCodeSnippet = (experimentId: string, scorersDocLink?: string) => `import mlflow
import os
from mlflow.genai import evaluate
from mlflow.genai.scorers import (
    Safety,
    RelevanceToQuery,
    Guidelines,
)

os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your API key
mlflow.set_experiment(experiment_id="${experimentId}")

# Step 1: Define evaluation dataset
eval_dataset = [{
  "inputs": {
    "query": "What is MLflow?",
  }
}]

# Step 2: Define predict_fn
# predict_fn will be called for every row in your evaluation
# dataset. Replace with your app's prediction function.
# NOTE: The **kwargs to predict_fn are the same as the keys of
# the \`inputs\` in your dataset.
def predict(query):
  return query + " an answer"

# Step 3: Run evaluation
# Select scorers relevant to your use case.${scorersDocLink ? `\n# See all available scorers: ${scorersDocLink}` : ''}
evaluate(
  data=eval_dataset,
  predict_fn=predict,
  scorers=[
    Safety(),
    RelevanceToQuery(),
    Guidelines(name="conciseness", guidelines="Responses must be concise."),
  ],
)

# Results will appear back in this UI`;

const getTraceCodeSnippet = (experimentId: string) => `import mlflow
import os
from mlflow.genai import evaluate
from mlflow.genai.scorers import (
    Safety,
    RelevanceToQuery,
    Guidelines,
)

os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your API key
mlflow.set_experiment(experiment_id="${experimentId}")

# Step 1: Pull traces to evaluate.
# Adjust max_results, or add a filter_string for time/status, etc.
# See: https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/
traces = mlflow.search_traces(max_results=20)

# Step 2: Run evaluation. No predict_fn needed — inputs/outputs
# are extracted from the trace objects automatically.
evaluate(
  data=traces,
  scorers=[
    Safety(),
    RelevanceToQuery(),
    Guidelines(name="conciseness", guidelines="Responses must be concise."),
  ],
)

# Results will appear back in this UI`;

const RUN_EVAL_MODAL_TAB_CODE_SNIPPET = 'code-snippet';
const RUN_EVAL_MODAL_TAB_TRACES = 'traces';

type JudgeSelectionMode = 'llm' | 'template';

export const RunEvaluationButton = ({ experimentId }: { experimentId: string }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<string>(RUN_EVAL_MODAL_TAB_CODE_SNIPPET);
  const [selectedTraceIds, setSelectedTraceIds] = useState<string[]>([]);
  const [isSelectTracesModalOpen, setIsSelectTracesModalOpen] = useState(false);
  const [selectedScorers, setSelectedScorers] = useState<LLMScorer[]>([]);
  const [selectedTemplates, setSelectedTemplates] = useState<LLM_TEMPLATE[]>([]);
  const [currentEndpointModel, setCurrentEndpointModel] = useState<string | undefined>(undefined);
  const hasSelectedTemplates = selectedTemplates.length > 0;
  const selectedJudgeCount = selectedScorers.length + selectedTemplates.length;
  const runJudgeDisabled = selectedJudgeCount === 0 || (hasSelectedTemplates && !currentEndpointModel);

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
  const evalInstructions = (
    <FormattedMessage
      defaultMessage="Run the following code to start an evaluation."
      description="Instructions for running the evaluation code in OSS"
    />
  );
  const { data: traceMetrics, isSuccess: isTraceMetricsLoaded } = useTraceMetricsQuery({
    experimentIds: [experimentId],
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    enabled: isOpen,
  });
  const traceCount = Number(traceMetrics?.data_points?.[0]?.values?.[AggregationType.COUNT] ?? 0);
  const hasTraces = traceCount > 0;
  const codeSnippet = hasTraces ? getTraceCodeSnippet(experimentId) : getDatasetCodeSnippet(experimentId);

  const traceSearchLocations = useMemo(() => [createTraceLocationForExperiment(experimentId)], [experimentId]);
  const { data: traceInfos, isLoading: isLoadingTraces } = useSearchMlflowTraces({
    locations: traceSearchLocations,
    disabled: !isOpen,
  });

  const hasSeededSelectionRef = useRef(false);
  useEffect(() => {
    if (hasSeededSelectionRef.current || !isOpen || isLoadingTraces || !traceInfos) {
      return;
    }
    hasSeededSelectionRef.current = true;
    setSelectedTraceIds(traceInfos.map((trace) => trace.trace_id));
  }, [isOpen, isLoadingTraces, traceInfos]);

  const hasSeededDefaultTabRef = useRef(false);
  useEffect(() => {
    if (!isOpen) {
      hasSeededDefaultTabRef.current = false;
      return;
    }
    if (hasSeededDefaultTabRef.current || !isTraceMetricsLoaded) {
      return;
    }
    hasSeededDefaultTabRef.current = true;
    setActiveTab(hasTraces ? RUN_EVAL_MODAL_TAB_TRACES : RUN_EVAL_MODAL_TAB_CODE_SNIPPET);
  }, [isOpen, isTraceMetricsLoaded, hasTraces]);

  const evalCodeSnippet = (
    <div css={{ position: 'relative' }}>
      <SnippetCopyAction
        componentId="mlflow.eval-runs.start-run-modal.copy-snippet"
        copyText={codeSnippet}
        css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
      />
      <CodeSnippet theme={theme.isDarkMode ? 'duotoneDark' : 'light'} language="python">
        {codeSnippet}
      </CodeSnippet>
    </div>
  );

  return (
    <>
      <Button componentId="mlflow.eval-runs.start-run-button" icon={<ChartLineIcon />} onClick={() => setIsOpen(true)}>
        <FormattedMessage
          defaultMessage="Run evaluation"
          description="Label for a button that displays instructions for starting a new evaluation run"
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
        okButtonProps={{ disabled: runJudgeDisabled }}
        onOk={() => {}}
        onCancel={() => setIsOpen(false)}
        footer={activeTab === RUN_EVAL_MODAL_TAB_TRACES ? undefined : null}
      >
        {!isTraceMetricsLoaded ? (
          // Wait for the trace count before rendering the tabs — otherwise the user
          // briefly sees only the Code Snippet tab and then watches it shift once the
          // trace tab pops in and we auto-seed activeTab to it.
          <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
            <Spinner />
          </div>
        ) : (
          <Tabs.Root componentId="mlflow.eval-runs.start-run-modal.tabs" value={activeTab} onValueChange={setActiveTab}>
            <Tabs.List>
              {hasTraces && (
                <Tabs.Trigger value={RUN_EVAL_MODAL_TAB_TRACES}>
                  <FormattedMessage
                    defaultMessage="Start an Evaluation"
                    description="Label for the trace-selection tab in the run evaluation modal"
                  />
                </Tabs.Trigger>
              )}
              <Tabs.Trigger value={RUN_EVAL_MODAL_TAB_CODE_SNIPPET}>
                <FormattedMessage
                  defaultMessage="Code Snippet"
                  description="Label for the code snippet tab in the run evaluation modal"
                />
              </Tabs.Trigger>
            </Tabs.List>
            {hasTraces && (
              <Tabs.Content value={RUN_EVAL_MODAL_TAB_TRACES}>
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
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
              </Tabs.Content>
            )}
            <Tabs.Content value={RUN_EVAL_MODAL_TAB_CODE_SNIPPET}>
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                <Typography.Text>{evalInstructions}</Typography.Text>
                {evalCodeSnippet}
              </div>
            </Tabs.Content>
          </Tabs.Root>
        )}
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
  selectedScorers,
  selectedTemplates,
  onToggleScorer,
  onToggleTemplate,
}: {
  experimentId: string;
  enabled: boolean;
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
  const { templateOptions } = useTemplateOptions(ScorerEvaluationScope.TRACES);

  const displayedLLMScorers = useMemo<LLMScorer[]>(() => {
    const lowercased = searchValue.toLowerCase();
    return ((data?.scheduledScorers ?? []) as LLMScorer[]).filter(
      (scorer) =>
        scorer.type === 'llm' && !scorer.isSessionLevelScorer && scorer.name.toLowerCase().includes(lowercased),
    );
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
                onClick={() => onToggleScorer(scorer)}
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
                onClick={() => onToggleTemplate(template.value)}
              />
            ))
          ))}
      </div>
    </div>
  );
};

const visualCheckboxCss = { pointerEvents: 'none' } as const;

const JudgeScorerOption = ({
  scorer,
  selected,
  onClick,
}: {
  scorer: LLMScorer;
  selected: boolean;
  onClick: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="checkbox"
      aria-checked={selected}
      css={{ cursor: 'pointer', height: 48, flexShrink: 0 }}
      onClick={onClick}
    >
      <div css={visualCheckboxCss}>
        <Checkbox componentId="mlflow.eval-runs.start-run-modal.judge-llm" isChecked={selected}>
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
    </div>
  );
};

const JudgeTemplateOption = ({
  template,
  selected,
  onClick,
}: {
  template: { value: LLM_TEMPLATE; label: string; hint: string };
  selected: boolean;
  onClick: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="checkbox"
      aria-checked={selected}
      css={{ cursor: 'pointer', height: 48, flexShrink: 0 }}
      onClick={onClick}
    >
      <div css={visualCheckboxCss}>
        <Checkbox componentId="mlflow.eval-runs.start-run-modal.judge-template" isChecked={selected}>
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
    </div>
  );
};

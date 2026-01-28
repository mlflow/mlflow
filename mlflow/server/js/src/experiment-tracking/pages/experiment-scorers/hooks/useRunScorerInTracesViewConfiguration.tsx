import { useCallback, useMemo, useRef, useState } from 'react';
import { LLM_TEMPLATE, LLMScorer } from '../types';
import { useGetScheduledScorers } from './useGetScheduledScorers';
import { useExperimentIds } from '../../../components/experiment-page/hooks/useExperimentIds';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  Alert,
  Button,
  getShadowScrollStyles,
  Input,
  Modal,
  PlusIcon,
  Radio,
  SearchIcon,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { PillControl } from '@databricks/design-system/development';
import ScorerModalRenderer from '../ScorerModalRenderer';
import { SCORER_FORM_MODE, ScorerEvaluationScope } from '../constants';
import { useRunSerializedScorer } from './useRunSerializedScorer';
import { ModelTraceExplorerRunJudgeConfig } from '@databricks/web-shared/model-trace-explorer';
import { ScorerFinishedEvent } from '../useEvaluateTracesAsync';
import { useTemplateOptions } from '../llmScorerUtils';
import { EndpointSelector } from '../../../components/EndpointSelector';
import {
  formatGatewayModelFromEndpoint,
  getEndpointNameFromGatewayModel,
} from '../../../../gateway/utils/gatewayUtils';
import { TEMPLATE_INSTRUCTIONS_MAP } from '../prompts';
import { isEmpty, isObject } from 'lodash';

interface UseRunScorerInTracesViewConfigurationReturnType extends ModelTraceExplorerRunJudgeConfig {
  RunJudgeModalElement: React.ReactNode;
}

export const useRunScorerInTracesViewConfiguration = (): UseRunScorerInTracesViewConfigurationReturnType => {
  const [experimentId] = useExperimentIds();

  const scorerFinishSubscribers = useRef<((event: ScorerFinishedEvent) => void)[]>([]);

  const onScorerFinished = useCallback((event: ScorerFinishedEvent) => {
    scorerFinishSubscribers.current.forEach((callback) => callback(event));
  }, []);

  const subscribeToScorerFinished = useCallback((callback: (event: ScorerFinishedEvent) => void) => {
    scorerFinishSubscribers.current.push(callback);
    return () => {
      scorerFinishSubscribers.current = scorerFinishSubscribers.current.filter(
        (currentCallback) => currentCallback !== callback,
      );
    };
  }, []);

  const { evaluateTraces, allEvaluations } = useRunSerializedScorer({ experimentId, onScorerFinished });

  const renderRunJudgeModal = useCallback<NonNullable<ModelTraceExplorerRunJudgeConfig['renderRunJudgeModal']>>(
    ({ traceId, onClose, visible }) => {
      return (
        <RunJudgeModalImpl visible={visible} traceId={traceId} evaluateTraces={evaluateTraces} onClose={onClose} />
      );
    },
    [evaluateTraces],
  );

  return {
    renderRunJudgeModal,
    evaluations: allEvaluations,
    subscribeToScorerFinished,
  } as UseRunScorerInTracesViewConfigurationReturnType;
};

/**
 * Dropdown for selecting a judge to run against a trace.
 */
const RunJudgeModalImpl = ({
  traceId,
  evaluateTraces,
  visible,
  onClose,
  scope = ScorerEvaluationScope.TRACES,
}: {
  traceId: string;
  evaluateTraces: (scorer: LLMScorer | LLM_TEMPLATE, traceIds: string[], endpointName?: string) => void;
  visible: boolean;
  onClose: () => void;
  scope?: ScorerEvaluationScope;
}) => {
  const [experimentId] = useExperimentIds();
  const { data, isLoading: loadingScorers } = useGetScheduledScorers(experimentId, { enabled: visible });
  const { templateOptions } = useTemplateOptions(scope);
  const intl = useIntl();

  const { theme } = useDesignSystemTheme();
  const [searchValue, setSearchValue] = useState<string>('');
  const [isCreateScorerModalVisible, setIsCreateScorerModalVisible] = useState(false);

  const [judgeSelectionMode, setJudgeSelectionMode] = useState<'llm' | 'template'>('llm');

  const [currentEndpointName, setCurrentEndpointName] = useState<string | undefined>(undefined);

  const displayedLLMScorers = useMemo(() => {
    return data?.scheduledScorers.filter(
      (scorer) => scorer.type === 'llm' && scorer.name.toLowerCase().includes(searchValue.toLowerCase()),
    ) as LLMScorer[];
  }, [data?.scheduledScorers, searchValue]);

  const displayedTemplates = useMemo(() => {
    // We don't support custom judges or guidelines templates in the traces view.
    const disabledTemplates = [LLM_TEMPLATE.CUSTOM, LLM_TEMPLATE.GUIDELINES];
    return templateOptions.filter(
      (template) =>
        !disabledTemplates.includes(template.value) && template.label.toLowerCase().includes(searchValue.toLowerCase()),
    );
  }, [templateOptions, searchValue]);

  const [error, setError] = useState<Error | undefined>(undefined);
  const [selectedJudge, setSelectedJudge] = useState<LLMScorer | LLM_TEMPLATE | undefined>(undefined);

  const handleModalConfirm = async () => {
    if (!selectedJudge) {
      return;
    }
    setError(undefined);
    try {
      evaluateTraces(selectedJudge, [traceId], currentEndpointName);
      onClose();
    } catch (error) {
      setError(error as Error);
    }
  };
  if (!visible) {
    return null;
  }

  return (
    <>
      <Modal
        componentId="mlflow.experiment-scorers.traces-view-judge-select-modal"
        visible
        onCancel={onClose}
        title={<FormattedMessage defaultMessage="Run judge on trace" description="Title for run judge modal" />}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Button text for canceling a judge run',
        })}
        okText={intl.formatMessage({
          defaultMessage: 'Run judge',
          description: 'Button text for running a judge',
        })}
        okButtonProps={{ disabled: !selectedJudge || (!currentEndpointName && judgeSelectionMode === 'template') }}
        onOk={handleModalConfirm}
      >
        <div css={{ display: 'flex', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}>
          <Input
            componentId="mlflow.experiment-scorers.traces-view-judge-search"
            prefix={<SearchIcon />}
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
            placeholder={intl.formatMessage({
              defaultMessage: 'Search judges',
              description: 'Placeholder for scorer search input',
            })}
          />
          <Button
            componentId="mlflow.experiment-scorers.traces-view-create-judge"
            icon={<PlusIcon />}
            onClick={() => setIsCreateScorerModalVisible(true)}
          >
            <FormattedMessage defaultMessage="Create judge" description="Button to create a new judge" />
          </Button>
        </div>
        <div css={{ marginBottom: theme.spacing.md }}>
          {error && (
            <Alert
              message={error.message}
              type="error"
              componentId="mlflow.experiment-scorers.traces-view-judge-error"
              css={{ marginBottom: theme.spacing.sm }}
              closable={false}
            />
          )}
          <PillControl.Root
            size="small"
            componentId="mlflow.experiment-scorers.traces-view-judge-type-filter"
            value={judgeSelectionMode}
            onValueChange={(value) => setJudgeSelectionMode(value as 'llm' | 'template')}
          >
            <PillControl.Item value="llm">
              <FormattedMessage
                defaultMessage="Custom LLM-as-a-judge ({llmCount})"
                description="Label for custom LLM judge type filter option"
                values={{ llmCount: displayedLLMScorers.length }}
              />
            </PillControl.Item>
            <PillControl.Item value="template">
              <FormattedMessage
                defaultMessage="Pre-built LLM-as-a-judge ({templateCount})"
                description="Label for pre-built LLM judge type filter option"
                values={{ templateCount: displayedTemplates.length }}
              />
            </PillControl.Item>
          </PillControl.Root>
        </div>
        <div
          css={{
            height: 240,
            display: 'flex',
            flexDirection: 'column',
            overflowY: 'auto',
            ...getShadowScrollStyles(theme, { orientation: 'vertical' }),
          }}
        >
          {judgeSelectionMode === 'llm' && (
            <>
              {loadingScorers && <TableSkeleton lines={3} />}
              {isEmpty(displayedLLMScorers) ? (
                <div css={{ display: 'flex', justifyContent: 'center' }}>
                  <Typography.Hint>
                    <FormattedMessage
                      defaultMessage="No custom LLM-as-a-judge scorers found"
                      description="Hint indicating that no custom LLM-as-a-judge scorers were found"
                    />
                  </Typography.Hint>
                </div>
              ) : (
                displayedLLMScorers?.map((scorer) => (
                  <ScorerOption
                    scorer={scorer}
                    key={scorer.name}
                    onClick={() => setSelectedJudge(scorer)}
                    selected={isObject(selectedJudge) && selectedJudge?.name === scorer.name}
                  />
                ))
              )}
            </>
          )}

          {judgeSelectionMode === 'template' &&
            displayedTemplates?.map((template) => (
              <TemplateOption
                selected={selectedJudge === template.value}
                template={template}
                key={template.value}
                onClick={() => setSelectedJudge(template.value)}
              />
            ))}
        </div>
        {judgeSelectionMode === 'template' && (
          <div
            css={{
              display: 'flex',
              marginTop: theme.spacing.sm,
              gap: theme.spacing.sm,
              flexDirection: 'column',
            }}
          >
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Endpoint:" description="Label for endpoint selection" />
            </Typography.Text>
            <EndpointSelector
              currentEndpointName={getEndpointNameFromGatewayModel(currentEndpointName)}
              onEndpointSelect={(endpointName) => {
                const modelValue = formatGatewayModelFromEndpoint(endpointName);
                setCurrentEndpointName(modelValue);
              }}
              autoSelectFirstEndpoint
            />
          </div>
        )}
      </Modal>
      {isCreateScorerModalVisible && (
        <ScorerModalRenderer
          visible
          onClose={() => setIsCreateScorerModalVisible(false)}
          experimentId={experimentId}
          mode={SCORER_FORM_MODE.CREATE}
          initialScorerType="llm"
        />
      )}
    </>
  );
};

const ScorerOption = ({
  scorer,
  onClick,
  selected,
}: {
  scorer: LLMScorer;
  onClick: (scorer: LLMScorer) => void;
  selected: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="radio"
      aria-checked={selected}
      css={{ cursor: 'pointer', height: 48, flexShrink: 0 }}
      onClick={() => onClick(scorer)}
    >
      <Radio componentId="mlflow.experiment-scorers.traces-view-judge-llm" checked={selected}>
        <div css={{ display: 'flex', flexDirection: 'column', marginLeft: theme.spacing.xs }}>
          <Typography.Text css={{ flex: 1 }}>{scorer.name}</Typography.Text>
          <Typography.Hint>
            <FormattedMessage defaultMessage="Custom judge" description="Label indicating a custom judge scorer" />
          </Typography.Hint>
        </div>
      </Radio>
    </div>
  );
};

const TemplateOption = ({
  template,
  onClick,
  selected,
}: {
  template: {
    value: LLM_TEMPLATE;
    label: string;
    hint: string;
  };
  onClick: (template: LLM_TEMPLATE) => void;
  selected: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="radio"
      aria-checked={selected}
      css={{ cursor: 'pointer', height: 48, flexShrink: 0 }}
      onClick={() => onClick(template.value)}
    >
      <Radio componentId="mlflow.experiment-scorers.traces-view-judge-template" checked={selected}>
        <div css={{ display: 'flex', flexDirection: 'column', marginLeft: theme.spacing.xs }}>
          <Typography.Text css={{ flex: 1 }}>{template.label}</Typography.Text>
          <Typography.Hint>
            {/* TODO: Add session level judges */}
            <FormattedMessage
              defaultMessage="Pre-built LLM-as-a-judge | Trace level"
              description="Label indicating a pre-built LLM-as-a-judge template"
            />
          </Typography.Hint>
        </div>
      </Radio>
    </div>
  );
};

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { LLMScorer } from '../types';
import { LLM_TEMPLATE } from '../types';
import { useGetScheduledScorers } from './useGetScheduledScorers';
import { useExperimentIds } from '../../../components/experiment-page/hooks/useExperimentIds';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  Alert,
  Button,
  PillControl,
  getShadowScrollStyles,
  Input,
  Modal,
  PlusIcon,
  Checkbox,
  SearchIcon,
  Spinner,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import ScorerModalRenderer from '../ScorerModalRenderer';
import { SCORER_FORM_MODE, ScorerEvaluationScope } from '../constants';
import { useRunSerializedScorer } from './useRunSerializedScorer';
import type { ModelTraceExplorerRunJudgeConfig } from '@databricks/web-shared/model-trace-explorer';
import type { ScorerEvaluation, ScorerFinishedEvent } from '../useEvaluateTracesAsync';
import { useTemplateOptions } from '../llmScorerUtils';
import { EndpointSelector } from '../../../components/EndpointSelector';
import {
  formatGatewayModelFromEndpoint,
  getEndpointNameFromGatewayModel,
} from '../../../../gateway/utils/gatewayUtils';
import { TEMPLATE_INSTRUCTIONS_MAP } from '../prompts';
import { isEmpty } from 'lodash';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { invalidateMlflowSearchTracesCache } from '../../../../shared/web-shared/model-trace-explorer/hooks/invalidateMlflowSearchTracesCache';

interface UseRunScorerInTracesViewConfigurationReturnType extends ModelTraceExplorerRunJudgeConfig {
  evaluateTraces: (scorer: LLMScorer | LLM_TEMPLATE, traceIds: string[], endpointName?: string) => void;
  allEvaluations: Record<string, ScorerEvaluation>;
}

export const useRunScorerInTracesViewConfiguration = (
  scope: ScorerEvaluationScope = ScorerEvaluationScope.TRACES,
): UseRunScorerInTracesViewConfigurationReturnType => {
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

  const { evaluateTraces, allEvaluations, reset } = useRunSerializedScorer({
    experimentId,
    onScorerFinished,
    scope,
  });

  const renderRunJudgeModal = useCallback<NonNullable<ModelTraceExplorerRunJudgeConfig['renderRunJudgeModal']>>(
    ({ itemId, onClose, visible }) => {
      return (
        <RunJudgeModalImpl
          scope={scope}
          visible={visible}
          itemIds={[itemId]}
          evaluateTraces={evaluateTraces}
          onClose={onClose}
        />
      );
    },
    [evaluateTraces, scope],
  );

  return {
    renderRunJudgeModal,
    evaluations: allEvaluations as ModelTraceExplorerRunJudgeConfig['evaluations'],
    evaluateTraces,
    allEvaluations: allEvaluations ?? {},
    subscribeToScorerFinished:
      subscribeToScorerFinished as ModelTraceExplorerRunJudgeConfig['subscribeToScorerFinished'],
    reset,
    scope,
  };
};

/**
 * Returns helpers for running judges on multiple selected traces from the table toolbar.
 * Accepts `evaluateTraces` and `allEvaluations` from a shared scorer instance so that
 * bulk evaluations are visible in `ModelTraceExplorerRunJudgesContext` (used by AssessmentCell).
 */
export const useRunJudgesOnTracesConfiguration = (
  evaluateTraces: (scorer: LLMScorer | LLM_TEMPLATE, traceIds: string[], endpointName?: string) => void,
  allEvaluations: Record<string, ScorerEvaluation> | undefined,
  subscribeToScorerFinished?: ModelTraceExplorerRunJudgeConfig['subscribeToScorerFinished'],
  scope: ScorerEvaluationScope = ScorerEvaluationScope.TRACES,
) => {
  const queryClient = useQueryClient();
  const [pendingTraceIds, setPendingTraceIds] = useState<string[]>([]);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [dismissedKeys, setDismissedKeys] = useState<Set<string>>(new Set());

  const showRunJudgesModal = useCallback((traceIds: string[]) => {
    setPendingTraceIds(traceIds);
    setIsModalVisible(true);
  }, []);

  const handleClose = useCallback(() => {
    setIsModalVisible(false);
    setPendingTraceIds([]);
  }, []);

  // Prune dismissed keys that no longer exist in allEvaluations to prevent unbounded growth
  useEffect(() => {
    if (!allEvaluations) return;
    const currentKeys = new Set(Object.values(allEvaluations).map((e) => e.requestKey));
    setDismissedKeys((prev) => {
      const pruned = new Set([...prev].filter((key) => currentKeys.has(key)));
      return pruned.size === prev.size ? prev : pruned;
    });
  }, [allEvaluations]);

  const activeEvaluations = useMemo(
    () => Object.values(allEvaluations ?? {}).filter((e) => !dismissedKeys.has(e.requestKey)),
    [allEvaluations, dismissedKeys],
  );

  const dismiss = useCallback((requestKey: string) => {
    setDismissedKeys((prev) => new Set([...prev, requestKey]));
  }, []);

  const JudgesStatusBanner = useMemo(
    () =>
      activeEvaluations.length > 0 ? (
        <JudgesEvaluationStatusBanner evaluations={activeEvaluations} onDismiss={dismiss} />
      ) : null,
    [activeEvaluations, dismiss],
  );

  const RunJudgesModal = useMemo(
    () => (
      <RunJudgeModalImpl
        scope={scope}
        visible={isModalVisible}
        itemIds={pendingTraceIds}
        evaluateTraces={evaluateTraces}
        onClose={handleClose}
      />
    ),
    [scope, isModalVisible, pendingTraceIds, evaluateTraces, handleClose],
  );

  // Invalidate the traces search cache when a scorer finishes
  useEffect(() => {
    return subscribeToScorerFinished?.(() => {
      invalidateMlflowSearchTracesCache({ queryClient });
    });
  }, [subscribeToScorerFinished, queryClient]);

  return { showRunJudgesModal, RunJudgesModal, JudgesStatusBanner };
};

/**
 * Dropdown for selecting a judge to run against one or more traces.
 */
const RunJudgeModalImpl = ({
  itemIds,
  evaluateTraces,
  visible,
  onClose,
  scope = ScorerEvaluationScope.TRACES,
}: {
  itemIds: string[];
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
    const isDisplayingSessionLevelScorers = scope === ScorerEvaluationScope.SESSIONS;
    return data?.scheduledScorers.filter(
      (scorer) =>
        scorer.type === 'llm' &&
        (scorer.isSessionLevelScorer ?? false) === isDisplayingSessionLevelScorers &&
        scorer.name.toLowerCase().includes(searchValue.toLowerCase()),
    ) as LLMScorer[];
  }, [data?.scheduledScorers, searchValue, scope]);

  const displayedTemplates = useMemo(() => {
    // We don't support custom judges or guidelines templates in the traces view.
    const disabledTemplates = [LLM_TEMPLATE.CUSTOM, LLM_TEMPLATE.GUIDELINES];
    return templateOptions.filter(
      (template) =>
        !disabledTemplates.includes(template.value) && template.label.toLowerCase().includes(searchValue.toLowerCase()),
    );
  }, [templateOptions, searchValue]);

  const [error, setError] = useState<Error | undefined>(undefined);
  const [selectedScorers, setSelectedScorers] = useState<LLMScorer[]>([]);
  const [selectedTemplates, setSelectedTemplates] = useState<LLM_TEMPLATE[]>([]);

  const selectedJudgeCount = selectedScorers.length + selectedTemplates.length;
  const hasSelectedTemplates = selectedTemplates.length > 0;

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

  const handleModalConfirm = async () => {
    if (selectedJudgeCount === 0) {
      return;
    }
    setError(undefined);
    try {
      selectedScorers.forEach((scorer) => evaluateTraces(scorer, itemIds, currentEndpointName));
      selectedTemplates.forEach((template) => evaluateTraces(template, itemIds, currentEndpointName));
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
        title={
          scope === ScorerEvaluationScope.SESSIONS ? (
            <FormattedMessage
              defaultMessage="Run judge on session"
              description="Title for run judge modal in sessions view"
            />
          ) : itemIds.length > 1 ? (
            <FormattedMessage
              defaultMessage="Run judge on {count} traces"
              description="Title for run judge modal when running on multiple traces"
              values={{ count: itemIds.length }}
            />
          ) : (
            <FormattedMessage
              defaultMessage="Run judge on trace"
              description="Title for run judge modal in traces view"
            />
          )
        }
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Button text for canceling a judge run',
        })}
        okText={
          selectedJudgeCount > 1
            ? intl.formatMessage({
                defaultMessage: 'Run judges',
                description: 'Button text for running multiple judges',
              })
            : intl.formatMessage({
                defaultMessage: 'Run judge',
                description: 'Button text for running a judge',
              })
        }
        okButtonProps={{
          disabled: selectedJudgeCount === 0 || (hasSelectedTemplates && !currentEndpointName),
        }}
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
                values={{ llmCount: displayedLLMScorers?.length ?? 0 }}
              />
            </PillControl.Item>
            <PillControl.Item value="template">
              <FormattedMessage
                defaultMessage="Pre-built LLM-as-a-judge ({templateCount})"
                description="Label for pre-built LLM judge type filter option"
                values={{ templateCount: displayedTemplates?.length ?? 0 }}
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
                    onClick={() => toggleScorer(scorer)}
                    selected={selectedScorers.some((s) => s.name === scorer.name)}
                  />
                ))
              )}
            </>
          )}

          {judgeSelectionMode === 'template' &&
            displayedTemplates?.map((template) => (
              <TemplateOption
                selected={selectedTemplates.includes(template.value)}
                template={template}
                key={template.value}
                onClick={() => toggleTemplate(template.value)}
                scope={scope}
              />
            ))}
        </div>
        {hasSelectedTemplates && (
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
          initialScope={scope}
          initialItemId={itemIds[0]}
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
      role="checkbox"
      aria-checked={selected}
      css={{ cursor: 'pointer', height: 48, flexShrink: 0 }}
      onClick={() => onClick(scorer)}
    >
      <Checkbox componentId="mlflow.experiment-scorers.traces-view-judge-llm" isChecked={selected}>
        <div css={{ display: 'flex', flexDirection: 'column', marginLeft: theme.spacing.xs }}>
          <Typography.Text css={{ flex: 1 }}>{scorer.name}</Typography.Text>
          <Typography.Hint>
            <FormattedMessage defaultMessage="Custom judge" description="Label indicating a custom judge scorer" />
          </Typography.Hint>
        </div>
      </Checkbox>
    </div>
  );
};

const TemplateOption = ({
  template,
  onClick,
  selected,
  scope,
}: {
  template: {
    value: LLM_TEMPLATE;
    label: string;
    hint: string;
  };
  onClick: (template: LLM_TEMPLATE) => void;
  selected: boolean;
  scope: ScorerEvaluationScope;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="checkbox"
      aria-checked={selected}
      css={{ cursor: 'pointer', height: 48, flexShrink: 0 }}
      onClick={() => onClick(template.value)}
    >
      <Checkbox componentId="mlflow.experiment-scorers.traces-view-judge-template" isChecked={selected}>
        <div css={{ display: 'flex', flexDirection: 'column', marginLeft: theme.spacing.xs }}>
          <Typography.Text css={{ flex: 1 }}>{template.label}</Typography.Text>
          <Typography.Hint>
            {scope === ScorerEvaluationScope.SESSIONS ? (
              <FormattedMessage
                defaultMessage="Pre-built LLM-as-a-judge | Session level"
                description="Label indicating a pre-built session-level LLM-as-a-judge template"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Pre-built LLM-as-a-judge | Trace level"
                description="Label indicating a pre-built trace-level LLM-as-a-judge template"
              />
            )}
          </Typography.Hint>
        </div>
      </Checkbox>
    </div>
  );
};

/**
 * A banner shown below the traces toolbar while judges are being evaluated
 * on selected traces. Shows one row per in-flight or completed evaluation,
 * with a spinner while loading and success/error states on completion.
 */
const AUTO_DISMISS_DELAY_MS = 10000;
const FADE_OUT_DURATION_MS = 1000;
const completedEvaluationFadeOutCss = {
  animation: `fadeOut ${FADE_OUT_DURATION_MS}ms ease-out ${AUTO_DISMISS_DELAY_MS - FADE_OUT_DURATION_MS}ms forwards`,
  '@keyframes fadeOut': { from: { opacity: 1 }, to: { opacity: 0 } },
} as const;

const JudgesEvaluationStatusBanner = ({
  evaluations,
  onDismiss,
}: {
  evaluations: ScorerEvaluation[];
  onDismiss: (requestKey: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  useEffect(() => {
    const completedKeys = new Set(evaluations.filter((e) => !e.isLoading).map((e) => e.requestKey));

    // Start timers for newly completed evaluations that don't have a timer yet
    completedKeys.forEach((key) => {
      if (!timersRef.current.has(key)) {
        const timer = setTimeout(() => {
          onDismiss(key);
          timersRef.current.delete(key);
        }, AUTO_DISMISS_DELAY_MS);
        timersRef.current.set(key, timer);
      }
    });

    // Clear timers for evaluations no longer in the list (e.g., already dismissed)
    timersRef.current.forEach((timer, key) => {
      if (!completedKeys.has(key)) {
        clearTimeout(timer);
        timersRef.current.delete(key);
      }
    });
  }, [evaluations, onDismiss]);

  // Clean up all pending timers on unmount
  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      timers.forEach(clearTimeout);
    };
  }, []);

  return (
    <div
      css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, padding: `0 0 ${theme.spacing.xs}px` }}
    >
      {evaluations.map((evaluation) => {
        if (evaluation.isLoading) {
          return (
            <Alert
              key={evaluation.requestKey}
              componentId="mlflow.experiment-scorers.judges-running-banner"
              type="info"
              closable={false}
              message={
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                  <Spinner size="small" />
                  <span>
                    {intl.formatMessage(
                      {
                        defaultMessage: 'Running judge "{label}"…',
                        description: 'Banner message shown while a judge is running on selected traces',
                      },
                      { label: evaluation.label },
                    )}
                  </span>
                </div>
              }
            />
          );
        }

        if (evaluation.error) {
          return (
            <div key={evaluation.requestKey} css={completedEvaluationFadeOutCss}>
              <Alert
                componentId="mlflow.experiment-scorers.judges-error-banner"
                type="error"
                closable
                onClose={() => onDismiss(evaluation.requestKey)}
                message={intl.formatMessage(
                  {
                    defaultMessage: 'Judge "{label}" failed: {error}',
                    description: 'Banner message shown when a judge run fails',
                  },
                  { label: evaluation.label, error: evaluation.error.message },
                )}
              />
            </div>
          );
        }

        return (
          <div key={evaluation.requestKey} css={completedEvaluationFadeOutCss}>
            <Alert
              componentId="mlflow.experiment-scorers.judges-success-banner"
              type="info"
              closable
              onClose={() => onDismiss(evaluation.requestKey)}
              message={intl.formatMessage(
                {
                  defaultMessage: 'Judge "{label}" completed successfully.',
                  description: 'Banner message shown when a judge run completes successfully',
                },
                { label: evaluation.label },
              )}
            />
          </div>
        );
      })}
    </div>
  );
};

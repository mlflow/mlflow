import React, { useMemo } from 'react';
import {
  useDesignSystemTheme,
  Typography,
  Button,
  PlayCircleFillIcon,
  LoadingState,
  ChevronLeftIcon,
  ChevronRightIcon,
  Alert,
  Tooltip,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { SimplifiedModelTraceExplorer } from '@databricks/web-shared/model-trace-explorer';
import type { Assessment, ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { COMPONENT_ID_PREFIX, BUTTON_VARIANT, type ButtonVariant, ScorerEvaluationScope } from './constants';
import { EvaluateTracesParams } from './types';
import { SampleScorerTracesToEvaluatePicker } from './SampleScorerTracesToEvaluatePicker';
import { useFormContext } from 'react-hook-form';
import { ScorerFormData } from './utils/scorerTransformUtils';
import { coerceToEnum } from '../../../shared/web-shared/utils';
import { ExperimentSingleChatConversation } from '../experiment-chat-sessions/single-chat-view/ExperimentSingleChatConversation';
import { SimplifiedAssessmentView } from '../../../shared/web-shared/model-trace-explorer/right-pane/SimplifiedAssessmentView';
import { compact } from 'lodash';
import { isSessionJudgeEvaluationResult, JudgeEvaluationResult } from './useEvaluateTraces.common';

/**
 * Run scorer button component.
 * Handles both "Run scorer" and "Re-Run scorer" variants with appropriate styling.
 */
const RunScorerButton: React.FC<{
  // Dummy comment to ensure copybara won't fail with formatting issues
  variant: ButtonVariant;
  onClick: () => Promise<void>;
  loading: boolean;
  disabled: boolean;
}> = ({
  // Dummy comment to ensure copybara won't fail with formatting issues
  variant,
  onClick,
  loading,
  disabled,
}) => {
  const { theme } = useDesignSystemTheme();
  const isRerun = variant === BUTTON_VARIANT.RERUN;

  const button = (
    <Button
      componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_52"
      type="primary"
      size={isRerun ? 'small' : undefined}
      onClick={onClick}
      loading={loading}
      disabled={disabled}
    >
      <PlayCircleFillIcon css={{ marginRight: isRerun ? theme.spacing.xs : theme.spacing.sm }} />
      {isRerun ? (
        <FormattedMessage defaultMessage="Re-Run judge" description="Button text for re-running judge" />
      ) : (
        <FormattedMessage defaultMessage="Run judge" description="Button text for running judge" />
      )}
    </Button>
  );
  return button;
};

interface SampleScorerOutputPanelRendererProps {
  isLoading: boolean;
  isRunScorerDisabled: boolean;
  runScorerDisabledTooltip?: string;
  error: Error | null;
  currentEvalResultIndex: number;
  currentEvalResult?: JudgeEvaluationResult;
  assessments: Assessment[] | undefined;
  handleRunScorer: () => Promise<void>;
  handlePrevious: () => void;
  handleNext: () => void;
  totalTraces: number;
  itemsToEvaluate: Pick<EvaluateTracesParams, 'itemCount' | 'itemIds'>;
  onItemsToEvaluateChange: (itemsToEvaluate: Pick<EvaluateTracesParams, 'itemCount' | 'itemIds'>) => void;
}

const SampleScorerOutputPanelRenderer: React.FC<SampleScorerOutputPanelRendererProps> = ({
  isLoading,
  isRunScorerDisabled,
  runScorerDisabledTooltip,
  error,
  currentEvalResultIndex,
  currentEvalResult,
  assessments,
  handleRunScorer,
  handlePrevious,
  handleNext,
  totalTraces,
  itemsToEvaluate,
  onItemsToEvaluateChange,
}) => {
  const { theme } = useDesignSystemTheme();

  // Whether we are showing a trace or the initial screen
  const isInitialScreen = !currentEvalResult;

  const { watch } = useFormContext<ScorerFormData>();
  const evaluationScope = coerceToEnum(ScorerEvaluationScope, watch('evaluationScope'), ScorerEvaluationScope.TRACES);

  // For session-level judges, get the traces from the current evaluation result
  const currentSessionTraces = useMemo(() => {
    if (!currentEvalResult || !isSessionJudgeEvaluationResult(currentEvalResult)) {
      return [];
    }
    return compact(currentEvalResult.traces?.map((trace) => trace));
  }, [currentEvalResult]);

  // Render the current evaluation result, either a trace or a chat session
  const renderCurrentEvaluationResult = () => {
    if (!currentEvalResult) {
      return null;
    }
    if (isSessionJudgeEvaluationResult(currentEvalResult)) {
      return (
        <div css={{ display: 'flex', gap: theme.spacing.md, paddingBottom: theme.spacing.md }}>
          <div css={{ flex: 1 }}>
            <ExperimentSingleChatConversation
              traces={currentSessionTraces}
              selectedTurnIndex={null}
              getAssessmentTitle={(assessmentName) => assessmentName}
            />
          </div>
          <div css={{ flex: 1 }}>
            <SimplifiedAssessmentView assessments={assessments ?? []} />
          </div>
        </div>
      );
    }
    if (!currentEvalResult.trace) {
      return null;
    }
    return <SimplifiedModelTraceExplorer modelTrace={currentEvalResult.trace} assessments={assessments ?? []} />;
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
      }}
    >
      {/* Header with title and dropdown */}
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          backgroundColor: theme.colors.backgroundSecondary,
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <Typography.Text bold css={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <FormattedMessage defaultMessage="Sample judge output" description="Title for sample judge output panel" />
        </Typography.Text>
        <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
          <SampleScorerTracesToEvaluatePicker
            itemsToEvaluate={itemsToEvaluate}
            onItemsToEvaluateChange={onItemsToEvaluateChange}
          />
          {!isInitialScreen && (
            <Tooltip
              componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_178"
              content={isRunScorerDisabled ? runScorerDisabledTooltip : undefined}
            >
              <span>
                <RunScorerButton
                  variant={BUTTON_VARIANT.RERUN}
                  onClick={handleRunScorer}
                  loading={isLoading}
                  disabled={isRunScorerDisabled}
                />
              </span>
            </Tooltip>
          )}
        </div>
      </div>
      {/* Content area */}
      <div
        css={{
          flex: 1,
          display: 'flex',
          minHeight: 0,
          backgroundColor: theme.colors.backgroundPrimary,
          overflowY: 'auto',
        }}
      >
        {!isInitialScreen && currentEvalResult ? (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              padding: theme.spacing.md,
              gap: theme.spacing.xs,
              flex: 1,
            }}
          >
            {/* Carousel controls and trace info */}
            <div
              css={{
                display: 'flex',
                justifyContent: 'flex-end',
                alignItems: 'center',
              }}
            >
              <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
                <Typography.Text size="sm" color="secondary">
                  <FormattedMessage
                    defaultMessage="{isTraces, select, true {Trace} other {Session}} {index} of {total}"
                    description="Index of the current trace and total number of traces"
                    values={{
                      index: currentEvalResultIndex + 1,
                      total: totalTraces,
                      isTraces: evaluationScope === ScorerEvaluationScope.TRACES,
                    }}
                  />
                </Typography.Text>
                <Button
                  componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_224"
                  size="small"
                  onClick={handlePrevious}
                  disabled={currentEvalResultIndex === 0}
                >
                  <ChevronLeftIcon />
                  <FormattedMessage defaultMessage="Previous" description="Button text for previous trace" />
                </Button>
                <Button
                  componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_234"
                  size="small"
                  onClick={handleNext}
                  disabled={currentEvalResultIndex === totalTraces - 1}
                >
                  <FormattedMessage defaultMessage="Next" description="Button text for next trace" />
                  <ChevronRightIcon />
                </Button>
              </div>
            </div>

            <div css={{ height: '600px' }}>{renderCurrentEvaluationResult()}</div>
          </div>
        ) : error ? (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              flex: 1,
              gap: theme.spacing.md,
              padding: theme.spacing.lg,
            }}
          >
            <Alert
              componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_263"
              type="error"
              message={error.message}
              closable={false}
              css={{ width: '100%', maxWidth: '600px' }}
            />
            <Tooltip
              componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_271"
              content={isRunScorerDisabled ? runScorerDisabledTooltip : undefined}
            >
              <span>
                <RunScorerButton
                  variant={BUTTON_VARIANT.RUN}
                  onClick={handleRunScorer}
                  loading={isLoading}
                  disabled={isRunScorerDisabled}
                />
              </span>
            </Tooltip>
          </div>
        ) : (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              flex: 1,
              textAlign: 'center',
              padding: theme.spacing.lg,
            }}
          >
            {isLoading && (
              <div css={{ marginBottom: theme.spacing.md }}>
                <LoadingState />
              </div>
            )}
            <Typography.Text size="lg" color="secondary" bold css={{ margin: 0, marginBottom: theme.spacing.xs }}>
              <FormattedMessage
                defaultMessage="Run judge on {isTraces, select, true {traces} other {sessions}}"
                description="Title for running judge on traces or sessions"
                values={{ isTraces: evaluationScope === ScorerEvaluationScope.TRACES }}
              />
            </Typography.Text>
            <Typography.Text color="secondary" css={{ margin: 0, marginBottom: theme.spacing.md }}>
              <FormattedMessage
                defaultMessage="Run the judge on the selected group of {isTraces, select, true {traces} other {sessions}}"
                description="Description for running judge on traces or sessions"
                values={{ isTraces: evaluationScope === ScorerEvaluationScope.TRACES }}
              />
            </Typography.Text>
            <Tooltip
              componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_samplescoreroutputpanelrenderer_316"
              content={isRunScorerDisabled ? runScorerDisabledTooltip : undefined}
            >
              <span>
                <RunScorerButton
                  variant={BUTTON_VARIANT.RUN}
                  onClick={handleRunScorer}
                  loading={isLoading}
                  disabled={isRunScorerDisabled}
                />
              </span>
            </Tooltip>
          </div>
        )}
      </div>
    </div>
  );
};

export default SampleScorerOutputPanelRenderer;

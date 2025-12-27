import React from 'react';
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
import { COMPONENT_ID_PREFIX, BUTTON_VARIANT, type ButtonVariant } from './constants';
import { EvaluateTracesParams } from './types';
import { SampleScorerTracesToEvaluatePicker } from './SampleScorerTracesToEvaluatePicker';

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
      componentId={`${COMPONENT_ID_PREFIX}.${isRerun ? 'rerun-scorer-button' : 'run-scorer-button'}`}
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
  currentTraceIndex: number;
  currentTrace: ModelTrace | undefined;
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
  currentTraceIndex,
  currentTrace,
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
  const isInitialScreen = !currentTrace;

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
              componentId={`${COMPONENT_ID_PREFIX}.rerun-scorer-button-tooltip`}
              content={isRunScorerDisabled ? runScorerDisabledTooltip : undefined}
            >
              <RunScorerButton
                variant={BUTTON_VARIANT.RERUN}
                onClick={handleRunScorer}
                loading={isLoading}
                disabled={isRunScorerDisabled}
              />
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
        {!isInitialScreen && currentTrace ? (
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
                    defaultMessage="Trace {index} of {total}"
                    description="Index of the current trace and total number of traces"
                    values={{ index: currentTraceIndex + 1, total: totalTraces }}
                  />
                </Typography.Text>
                <Button
                  componentId={`${COMPONENT_ID_PREFIX}.previous-trace-button`}
                  size="small"
                  onClick={handlePrevious}
                  disabled={currentTraceIndex === 0}
                >
                  <ChevronLeftIcon />
                  <FormattedMessage defaultMessage="Previous" description="Button text for previous trace" />
                </Button>
                <Button
                  componentId={`${COMPONENT_ID_PREFIX}.next-trace-button`}
                  size="small"
                  onClick={handleNext}
                  disabled={currentTraceIndex === totalTraces - 1}
                >
                  <FormattedMessage defaultMessage="Next" description="Button text for next trace" />
                  <ChevronRightIcon />
                </Button>
              </div>
            </div>

            <div css={{ height: '600px' }}>
              <SimplifiedModelTraceExplorer modelTrace={currentTrace} assessments={assessments ?? []} />
            </div>
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
              componentId={`${COMPONENT_ID_PREFIX}.scorer-error-alert`}
              type="error"
              message={error.message}
              closable={false}
              css={{ width: '100%', maxWidth: '600px' }}
            />
            <Tooltip
              componentId={`${COMPONENT_ID_PREFIX}.run-scorer-button-error-tooltip`}
              content={isRunScorerDisabled ? runScorerDisabledTooltip : undefined}
            >
              <RunScorerButton
                variant={BUTTON_VARIANT.RUN}
                onClick={handleRunScorer}
                loading={isLoading}
                disabled={isRunScorerDisabled}
              />
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
              <FormattedMessage defaultMessage="Run judge on traces" description="Title for running judge on traces" />
            </Typography.Text>
            <Typography.Text color="secondary" css={{ margin: 0, marginBottom: theme.spacing.md }}>
              <FormattedMessage
                defaultMessage="Run the judge on the selected group of traces"
                description="Description for running judge on traces"
              />
            </Typography.Text>
            <Tooltip
              componentId={`${COMPONENT_ID_PREFIX}.run-scorer-button-initial-tooltip`}
              content={isRunScorerDisabled ? runScorerDisabledTooltip : undefined}
            >
              <RunScorerButton
                variant={BUTTON_VARIANT.RUN}
                onClick={handleRunScorer}
                loading={isLoading}
                disabled={isRunScorerDisabled}
              />
            </Tooltip>
          </div>
        )}
      </div>
    </div>
  );
};

export default SampleScorerOutputPanelRenderer;

import { isNil } from 'lodash';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import React, { useCallback, useEffect, useMemo, useState } from 'react';

import {
  ApplyDesignSystemContextOverrides,
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  GenericSkeleton,
  Modal,
  PlusIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import {
  isV3ModelTraceInfo,
  ModelTraceExplorer,
  ModelTraceExplorerSkeleton,
  shouldUseModelTraceExplorerDrawerUI,
  isV4TraceId,
  type ModelTrace,
} from '@databricks/web-shared/model-trace-explorer';

import { EvaluationsReviewDetailsHeader } from './EvaluationsReviewDetails';
import { GenAiEvaluationTracesReview } from './GenAiEvaluationTracesReview';
import { GenAITracesTableContext } from '../GenAITracesTableContext';
import { AssistantAwareDrawer } from '../../../../common/components/AssistantAwareDrawer';
import { useGenAITracesTableConfig } from '../hooks/useGenAITracesTableConfig';
import type { GetTraceFunction } from '../hooks/useGetTrace';
import { useGetTrace, useGetTraceByFullTraceId } from '../hooks/useGetTrace';
import type {
  AssessmentInfo,
  EvalTraceComparisonEntry,
  RunEvaluationTracesDataEntry,
  SaveAssessmentsQuery,
} from '../types';
import { shouldUseTracesV4API } from '../utils/FeatureUtils';
import { convertTraceInfoV3ToRunEvalEntry, getSpansLocation, TRACKING_STORE_SPANS_LOCATION } from '../utils/TraceUtils';

const MODAL_SPACING_REM = 4;
const DEFAULT_MODAL_MARGIN_REM = 1;

const evalEntryMatchesEvaluationId = (evaluationId: string, entry?: RunEvaluationTracesDataEntry) => {
  if (isV4TraceId(evaluationId) && entry?.fullTraceId === evaluationId) {
    return true;
  }
  return entry?.evaluationId === evaluationId;
};

export const GenAiEvaluationTracesReviewModal = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    experimentId,
    runUuid,
    evaluations,
    selectedEvaluationId,
    onChangeEvaluationId,
    runDisplayName,
    otherRunDisplayName,
    exportToEvalsInstanceEnabled = false,
    assessmentInfos,
    getTrace,
    saveAssessmentsQuery,
  }: {
    experimentId: string;
    runUuid?: string;
    evaluations: EvalTraceComparisonEntry[];
    selectedEvaluationId: string;
    onChangeEvaluationId: (evaluationId: string | undefined) => void;
    runDisplayName?: string;
    otherRunDisplayName?: string;
    exportToEvalsInstanceEnabled?: boolean;
    assessmentInfos: AssessmentInfo[];
    getTrace?: GetTraceFunction;
    saveAssessmentsQuery?: SaveAssessmentsQuery;
  }) => {
    const { theme } = useDesignSystemTheme();
    const [showAddToEvaluationDatasetModal, setShowAddToEvaluationDatasetModal] = useState(false);

    const handleClose = useCallback(() => {
      onChangeEvaluationId(undefined);
    }, [onChangeEvaluationId]);

    // The URL always has an evaluation id, so we look in either current or other for the eval.
    const findEval = useCallback(
      (entry: EvalTraceComparisonEntry) =>
        evalEntryMatchesEvaluationId(selectedEvaluationId, entry.currentRunValue) ||
        evalEntryMatchesEvaluationId(selectedEvaluationId, entry.otherRunValue),
      [selectedEvaluationId],
    );

    const previousEvaluationIdx = useMemo(
      () => (evaluations ? evaluations?.findIndex(findEval) - 1 : undefined),
      [evaluations, findEval],
    );
    const isPreviousAvailable = useMemo(
      () => previousEvaluationIdx !== undefined && previousEvaluationIdx >= 0,
      [previousEvaluationIdx],
    );

    const nextEvaluationIdx = useMemo(
      () => (evaluations ? evaluations?.findIndex(findEval) + 1 : undefined),
      [evaluations, findEval],
    );
    const isNextAvailable = useMemo(
      () => nextEvaluationIdx !== undefined && nextEvaluationIdx < evaluations.length,
      [nextEvaluationIdx, evaluations],
    );

    const selectPreviousEval = useCallback(() => {
      if (evaluations === null || previousEvaluationIdx === undefined) return;

      const newEvalId =
        evaluations[previousEvaluationIdx]?.currentRunValue?.evaluationId ||
        evaluations[previousEvaluationIdx]?.otherRunValue?.evaluationId;
      onChangeEvaluationId(newEvalId);
    }, [evaluations, previousEvaluationIdx, onChangeEvaluationId]);

    // prettier-ignore
    const {
      renderExportTracesToDatasetsModal,
    } = React.useContext(GenAITracesTableContext);

    const selectNextEval = useCallback(() => {
      if (evaluations === null || nextEvaluationIdx === undefined) return;

      const newEvalId =
        evaluations[nextEvaluationIdx]?.currentRunValue?.evaluationId ||
        evaluations[nextEvaluationIdx]?.otherRunValue?.evaluationId;
      onChangeEvaluationId(newEvalId);
    }, [evaluations, nextEvaluationIdx, onChangeEvaluationId]);

    const evaluation = useMemo(() => evaluations?.find(findEval), [evaluations, findEval]);
    const nextEvaluation = useMemo(
      () => (nextEvaluationIdx && evaluations ? evaluations?.[nextEvaluationIdx] : undefined),
      [evaluations, nextEvaluationIdx],
    );
    const previousEvaluation = useMemo(
      () => (previousEvaluationIdx && evaluations ? evaluations?.[previousEvaluationIdx] : undefined),
      [evaluations, previousEvaluationIdx],
    );

    const tracesTableConfig = useGenAITracesTableConfig();

    // --- Auto-polling until trace is complete if the backend supports returning partial spans ---
    const spansLocation = getSpansLocation(evaluation?.currentRunValue?.traceInfo);
    const shouldEnablePolling = spansLocation === TRACKING_STORE_SPANS_LOCATION;

    // prettier-ignore
    const traceQueryResult = useGetTrace(
      getTrace,
      evaluation?.currentRunValue?.traceInfo,
      shouldEnablePolling,
    );
    // prettier-ignore
    const compareToTraceQueryResult = useGetTrace(
      getTrace,
      evaluation?.otherRunValue?.traceInfo,
      shouldEnablePolling,
    );
    // In case that the selected evaluation is not provided upstream (but the list is loaded), we lazily fetch the full trace data here
    const shouldFetchTraceBySearchParamId = useMemo(
      () => Boolean(evaluations) && !evaluation && Boolean(selectedEvaluationId),
      [evaluations, evaluation, selectedEvaluationId],
    );

    const traceBySearchParamQueryResult = useGetTraceByFullTraceId(
      getTrace,
      shouldFetchTraceBySearchParamId ? selectedEvaluationId : undefined,
    );

    // Prefetching the next and previous traces to optimize performance
    // prettier-ignore
    useGetTrace(
      getTrace,
      nextEvaluation?.currentRunValue?.traceInfo,
    );
    // prettier-ignore
    useGetTrace(
      getTrace,
      previousEvaluation?.currentRunValue?.traceInfo,
    );

    // is true if only one of the two runs has a trace
    const isSingleTraceView = Boolean(evaluation?.currentRunValue) !== Boolean(evaluation?.otherRunValue);

    const currentTraceQueryResult = shouldFetchTraceBySearchParamId
      ? traceBySearchParamQueryResult
      : evalEntryMatchesEvaluationId(selectedEvaluationId, evaluation?.currentRunValue)
        ? traceQueryResult
        : compareToTraceQueryResult;

    if (isNil(evaluation) && !shouldFetchTraceBySearchParamId) {
      return <></>;
    }

    const renderModalTitle = () => {
      if (shouldFetchTraceBySearchParamId) {
        if (traceBySearchParamQueryResult.isLoading) {
          return (
            <GenericSkeleton
              css={{
                width: 200,
                height: theme.general.heightBase,
              }}
            />
          );
        }
        if (traceBySearchParamQueryResult.data?.info && isV3ModelTraceInfo(traceBySearchParamQueryResult.data?.info)) {
          const runEvalEntry = convertTraceInfoV3ToRunEvalEntry(traceBySearchParamQueryResult.data?.info);
          return <EvaluationsReviewDetailsHeader evaluationResult={runEvalEntry} />;
        }
      }
      return evaluation?.currentRunValue ? (
        <EvaluationsReviewDetailsHeader evaluationResult={evaluation.currentRunValue} />
      ) : evaluation?.otherRunValue ? (
        <EvaluationsReviewDetailsHeader evaluationResult={evaluation.otherRunValue} />
      ) : null;
    };

    // Define the content of the modal/drawer
    const content = (
      <>
        {/* Only show skeleton for the first fetch to avoid flickering when polling new spans */}
        {!shouldUseModelTraceExplorerDrawerUI() &&
          !currentTraceQueryResult.data &&
          currentTraceQueryResult.isFetching && (
            <GenericSkeleton
              label="Loading trace..."
              style={{
                // Size the width and height to fit the modal content area
                width: 'calc(100% - 45px)',
                height: 'calc(100% - 100px)',
                position: 'absolute',
                paddingRight: 500,
                zIndex: 2100,
                backgroundColor: theme.colors.backgroundPrimary,
              }}
            />
          )}
        {
          // Show ModelTraceExplorer only if there is no run to compare to and there's trace data.
          ((shouldFetchTraceBySearchParamId && traceBySearchParamQueryResult?.data) || isSingleTraceView) &&
          !isNil(currentTraceQueryResult.data) ? (
            <div css={{ height: 'calc(100% - 34px)', marginLeft: -theme.spacing.lg, marginRight: -theme.spacing.lg }}>
              {/* prettier-ignore */}
              <ModelTraceExplorerModalBody
                traceData={currentTraceQueryResult.data}
                showLoadingState={shouldUseModelTraceExplorerDrawerUI() && (currentTraceQueryResult.isFetching)}
              />
            </div>
          ) : (
            evaluation?.currentRunValue &&
            (shouldUseModelTraceExplorerDrawerUI() && currentTraceQueryResult.isFetching ? (
              <div css={{ marginLeft: -theme.spacing.lg, marginRight: -theme.spacing.lg }}>
                <ModelTraceExplorerSkeleton />
              </div>
            ) : (
              <div
                css={
                  shouldUseModelTraceExplorerDrawerUI() ? { overflow: 'auto', height: '100%' } : { display: 'contents' }
                }
              >
                <GenAiEvaluationTracesReview
                  experimentId={experimentId}
                  evaluation={evaluation.currentRunValue}
                  otherEvaluation={evaluation.otherRunValue}
                  selectNextEval={selectNextEval}
                  isNextAvailable={isNextAvailable}
                  css={{ flex: 1, overflow: 'hidden' }}
                  runUuid={runUuid}
                  isReadOnly={!tracesTableConfig.enableRunEvaluationWriteFeatures}
                  runDisplayName={runDisplayName}
                  compareToRunDisplayName={otherRunDisplayName}
                  exportToEvalsInstanceEnabled={exportToEvalsInstanceEnabled}
                  assessmentInfos={assessmentInfos}
                  traceQueryResult={traceQueryResult}
                  compareToTraceQueryResult={compareToTraceQueryResult}
                  saveAssessmentsQuery={saveAssessmentsQuery}
                />
              </div>
            ))
          )
        }
      </>
    );

    // Decide which wrapper to use based on feature flag
    const WrapperComponent = shouldUseModelTraceExplorerDrawerUI() ? DrawerWrapper : ModalWrapper;

    return (
      <WrapperComponent
        handleClose={handleClose}
        isNextAvailable={isNextAvailable}
        isPreviousAvailable={isPreviousAvailable}
        selectNextEval={selectNextEval}
        selectPreviousEval={selectPreviousEval}
        renderModalTitle={renderModalTitle}
        isLoading={currentTraceQueryResult.isFetching}
        onAddTraceToEvaluationDatasetClick={() => setShowAddToEvaluationDatasetModal(true)}
      >
        {content}
        {renderExportTracesToDatasetsModal?.({
          experimentId,
          visible: showAddToEvaluationDatasetModal,
          setVisible: setShowAddToEvaluationDatasetModal,
          selectedTraceInfos: evaluation?.currentRunValue?.traceInfo ? [evaluation.currentRunValue.traceInfo] : [],
        })}
      </WrapperComponent>
    );
  },
);

const ModalWrapper = ({
  selectPreviousEval,
  selectNextEval,
  isPreviousAvailable,
  isNextAvailable,
  renderModalTitle,
  handleClose,
  children,
  isLoading,
}: {
  children: React.ReactNode;
  selectPreviousEval: () => void;
  selectNextEval: () => void;
  isPreviousAvailable: boolean;
  isNextAvailable: boolean;
  renderModalTitle: () => React.ReactNode;
  handleClose: () => void;
  isLoading?: boolean;
}) => {
  const { theme, classNamePrefix } = useDesignSystemTheme();
  const useRadixModal = false;

  return (
    <div
      onKeyDown={(e) => {
        if (e.key === 'ArrowLeft') {
          selectPreviousEval();
        } else if (e.key === 'ArrowRight') {
          selectNextEval();
        }
      }}
    >
      <Modal
        componentId="mlflow.evaluations_review.modal"
        visible
        title={renderModalTitle()}
        onCancel={handleClose}
        size="wide"
        verticalSizing="maxed_out"
        css={{
          width: '100% !important',
          padding: useRadixModal ? undefined : `0 ${MODAL_SPACING_REM}rem !important`,
          [`& .${classNamePrefix}-modal-body`]: {
            flex: 1,
            paddingTop: 0,
          },
          [`& .${classNamePrefix}-modal-header`]: {
            paddingBottom: theme.spacing.sm,
          },
        }}
        footer={null} // Hide the footer
      >
        {children}
        <div
          css={{
            display: 'flex',
            justifyContent: 'flex-end',
            position: 'fixed',
            top: '50%',
            left: 0,
            zIndex: 2000,
            opacity: '.75',
            width: `${MODAL_SPACING_REM + DEFAULT_MODAL_MARGIN_REM}rem`,
            '&:hover': {
              opacity: '1.0',
            },
          }}
        >
          <div
            css={{
              backgroundColor: theme.colors.backgroundPrimary,
              borderRadius: theme.legacyBorders.borderRadiusMd,
              marginRight: theme.spacing.sm,
            }}
          >
            <Button
              disabled={!isPreviousAvailable}
              componentId="mlflow.evaluations_review.modal.previous_eval"
              icon={<ChevronLeftIcon />}
              onClick={() => selectPreviousEval()}
            />
          </div>
        </div>
        <div
          css={{
            display: 'flex',
            justifyContent: 'flex-start',
            position: 'fixed',
            top: '50%',
            right: 0,
            zIndex: 2000,
            width: `${MODAL_SPACING_REM + DEFAULT_MODAL_MARGIN_REM}rem`,
            opacity: '.75',
            '&:hover': {
              opacity: '1.0',
            },
          }}
        >
          <div
            css={{
              backgroundColor: theme.colors.backgroundPrimary,
              borderRadius: theme.legacyBorders.borderRadiusMd,
              marginLeft: theme.spacing.sm,
            }}
          >
            <Button
              disabled={!isNextAvailable}
              componentId="mlflow.evaluations_review.modal.next_eval"
              icon={<ChevronRightIcon />}
              onClick={(e) => selectNextEval()}
            />
          </div>
        </div>
      </Modal>
    </div>
  );
};

const DrawerWrapper = ({
  selectPreviousEval,
  selectNextEval,
  isPreviousAvailable,
  isNextAvailable,
  renderModalTitle,
  handleClose,
  children,
  isLoading,
  onAddTraceToEvaluationDatasetClick,
}: {
  children: React.ReactNode;
  selectPreviousEval: () => void;
  selectNextEval: () => void;
  isPreviousAvailable: boolean;
  isNextAvailable: boolean;
  renderModalTitle: () => React.ReactNode;
  handleClose: () => void;
  isLoading?: boolean;
  onAddTraceToEvaluationDatasetClick?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement) {
        if (e.target.role === 'tab') {
          return;
        }
        const tagName = e.target?.tagName?.toLowerCase();
        if (tagName === 'input' || tagName === 'textarea' || e.target.isContentEditable) {
          return;
        }
      }
      if (e.key === 'ArrowLeft' && isPreviousAvailable) {
        e.stopPropagation();
        selectPreviousEval();
      } else if (e.key === 'ArrowRight' && isNextAvailable) {
        e.stopPropagation();
        selectNextEval();
      }
    },
    [isPreviousAvailable, isNextAvailable, selectPreviousEval, selectNextEval],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  return (
    <AssistantAwareDrawer.Root
      open
      onOpenChange={(open) => {
        if (!open) {
          handleClose();
        }
      }}
    >
      <AssistantAwareDrawer.Content
        componentId="mlflow.evaluations_review.modal"
        width="80vw"
        title={
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <Button
              componentId="mlflow.evaluations_review.modal.previous_eval"
              disabled={!isPreviousAvailable}
              onClick={() => selectPreviousEval()}
            >
              <ChevronLeftIcon />
            </Button>
            <Button
              componentId="mlflow.evaluations_review.modal.next_eval"
              disabled={!isNextAvailable}
              onClick={() => selectNextEval()}
            >
              <ChevronRightIcon />
            </Button>
            <div css={{ flex: 1, overflow: 'hidden' }}>{renderModalTitle()}</div>
            {onAddTraceToEvaluationDatasetClick && (
              <Button
                componentId="mlflow.evaluations_review.modal.add_to_evaluation_dataset"
                onClick={() => onAddTraceToEvaluationDatasetClick?.()}
                icon={<PlusIcon />}
              >
                <FormattedMessage
                  defaultMessage="Add to dataset"
                  description="Button text for adding a trace to a evaluation dataset"
                />
              </Button>
            )}
          </div>
        }
        expandContentToFullHeight
        css={[
          {
            // Disable drawer's scroll to allow inner content to handle scrolling
            '&>div': {
              overflow: 'hidden',
            },
            '&>div:first-child': {
              paddingLeft: theme.spacing.md,
              paddingTop: 1,
              paddingBottom: 1,
              // Prevent close button from being squeezed
              '&>button': {
                flexShrink: 0,
              },
            },
          },
        ]}
      >
        <ApplyDesignSystemContextOverrides zIndexBase={2 * theme.options.zIndexBase}>
          {isLoading ? <ModelTraceExplorerSkeleton /> : <>{children}</>}
        </ApplyDesignSystemContextOverrides>
      </AssistantAwareDrawer.Content>
    </AssistantAwareDrawer.Root>
  );
};

// prettier-ignore
const ModelTraceExplorerModalBody = ({
  traceData,
  showLoadingState,
}: {
  traceData: ModelTrace;
  showLoadingState: boolean;
}) => {
  return (
    <ModelTraceExplorer
      modelTrace={traceData}
    />
  );
};

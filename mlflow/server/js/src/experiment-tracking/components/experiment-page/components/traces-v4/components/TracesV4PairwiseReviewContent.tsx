import { useCallback, useEffect, useState } from 'react';

import {
  Card,
  CheckboxIcon,
  Input,
  SegmentedControlButton,
  SegmentedControlGroup,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { RunColorCircle } from '@databricks/web-shared/genai-traces-table';
import {
  ModelTraceExplorer,
  ModelTraceExplorerUpdateTraceContextProvider,
  isV3ModelTraceInfo,
  type ModelTrace,
} from '@databricks/web-shared/model-trace-explorer';
import { useIntl } from 'react-intl';
import { KeyboardShortcutKey } from '../../../../../pages/experiment-review-queue/KeyboardShortcutKey';
import { usePairwiseReviewAssessmentMutation } from '../hooks/usePairwiseReviewAssessmentMutation';
import {
  findPairwiseScoreAssessment,
  getPairwiseScoreValue,
  PAIRWISE_SCORE_LOSE_VALUE,
  PAIRWISE_SCORE_WIN_VALUE,
} from '../utils/pairwiseReview';

export interface TracesV4PairwiseReviewContentProps {
  baselineTrace: ModelTrace;
  baselineTraceId: string;
  baselineRunName?: string;
  candidateTrace: ModelTrace;
  candidateTraceId: string;
  candidateRunName?: string;
  onEditTags?: (traceInfo: ModelTrace['info']) => void;
}

const getTraceIdPreview = (traceId: string): string => {
  const finalSegment = traceId.split('/').pop() ?? traceId;
  return finalSegment.replace(/^tr-/, '').slice(0, 8);
};

export const TracesV4PairwiseReviewContent = ({
  baselineTrace,
  baselineTraceId,
  baselineRunName,
  candidateTrace,
  candidateTraceId,
  candidateRunName,
  onEditTags,
}: TracesV4PairwiseReviewContentProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const baselineLabel = baselineRunName ?? getTraceIdPreview(baselineTraceId);
  const candidateLabel = candidateRunName ?? getTraceIdPreview(candidateTraceId);
  const baselineTraceInfo = isV3ModelTraceInfo(baselineTrace.info) ? baselineTrace.info : undefined;
  const candidateTraceInfo = isV3ModelTraceInfo(candidateTrace.info) ? candidateTrace.info : undefined;
  const {
    savePairwiseReview,
    isSavingPairwiseReview,
    reviewerSourceId: sourceId,
    isReviewerResolved,
  } = usePairwiseReviewAssessmentMutation();
  const isReviewLocked = isSavingPairwiseReview || !isReviewerResolved;
  const existingBaselineAssessment =
    baselineTraceInfo && candidateTraceInfo
      ? findPairwiseScoreAssessment({
          trace: baselineTraceInfo,
          opponentTraceId: candidateTraceInfo.trace_id,
          sourceId,
        })
      : undefined;
  const existingCandidateAssessment =
    baselineTraceInfo && candidateTraceInfo
      ? findPairwiseScoreAssessment({
          trace: candidateTraceInfo,
          opponentTraceId: baselineTraceInfo.trace_id,
          sourceId,
        })
      : undefined;
  const existingBaselineScore = existingBaselineAssessment
    ? getPairwiseScoreValue(existingBaselineAssessment)
    : undefined;
  const existingCandidateScore = existingCandidateAssessment
    ? getPairwiseScoreValue(existingCandidateAssessment)
    : undefined;
  const persistedSelectedTraceId =
    existingBaselineScore === PAIRWISE_SCORE_WIN_VALUE && existingCandidateScore === PAIRWISE_SCORE_LOSE_VALUE
      ? baselineTraceId
      : existingBaselineScore === PAIRWISE_SCORE_LOSE_VALUE && existingCandidateScore === PAIRWISE_SCORE_WIN_VALUE
        ? candidateTraceId
        : undefined;
  const persistedRationale = existingBaselineAssessment?.rationale ?? existingCandidateAssessment?.rationale ?? '';
  const [selectedTraceIdOverride, setSelectedTraceId] = useState<string>();
  const [rationaleOverride, setRationale] = useState<string>();
  const [lastSavedRationaleOverride, setLastSavedRationale] = useState<string>();
  const [showBaselineSpanTree, setShowBaselineSpanTree] = useState(false);
  const [showCandidateSpanTree, setShowCandidateSpanTree] = useState(false);
  const selectedTraceId = selectedTraceIdOverride ?? persistedSelectedTraceId;
  const rationale = rationaleOverride ?? persistedRationale;
  const lastSavedRationale = lastSavedRationaleOverride ?? persistedRationale;
  const questionValue =
    selectedTraceId === baselineTraceId ? true : selectedTraceId === candidateTraceId ? false : null;
  const commentLabel = intl.formatMessage({
    defaultMessage: 'Comment',
    description: 'Accessible label for the comment in pairwise trace review',
  });

  const persistReview = useCallback(
    async (winnerTraceId: string, nextRationale: string) => {
      if (!baselineTraceInfo || !candidateTraceInfo) {
        return;
      }
      const baselineWins = winnerTraceId === baselineTraceId;
      await savePairwiseReview({
        baseline: {
          traceInfo: baselineTraceInfo,
          opponentTraceId: candidateTraceInfo.trace_id,
          score: baselineWins ? PAIRWISE_SCORE_WIN_VALUE : PAIRWISE_SCORE_LOSE_VALUE,
        },
        candidate: {
          traceInfo: candidateTraceInfo,
          opponentTraceId: baselineTraceInfo.trace_id,
          score: baselineWins ? PAIRWISE_SCORE_LOSE_VALUE : PAIRWISE_SCORE_WIN_VALUE,
        },
        rationale: nextRationale,
      });
      setLastSavedRationale(nextRationale);
    },
    [baselineTraceId, baselineTraceInfo, candidateTraceInfo, savePairwiseReview],
  );

  const selectWinner = useCallback(
    (nextSelectedTraceId: string) => {
      if (!baselineTraceInfo || !candidateTraceInfo || isReviewLocked || nextSelectedTraceId === selectedTraceId) {
        return;
      }
      const previousSelectedTraceId = selectedTraceId;
      setSelectedTraceId(nextSelectedTraceId);
      void persistReview(nextSelectedTraceId, rationale).catch(() => {
        setSelectedTraceId(previousSelectedTraceId);
      });
    },
    [baselineTraceInfo, candidateTraceInfo, isReviewLocked, persistReview, rationale, selectedTraceId],
  );

  useEffect(() => {
    const handleWinnerShortcut = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.defaultPrevented) {
        return;
      }
      if (event.target instanceof HTMLElement) {
        const isEditableTarget =
          event.target.matches('textarea, select, input:not([type="radio"]):not([type="checkbox"])') ||
          event.target.isContentEditable ||
          Boolean(event.target.closest('[contenteditable="true"]'));
        if (isEditableTarget) {
          return;
        }
      }
      const winnerTraceId = event.key === '1' ? baselineTraceId : event.key === '2' ? candidateTraceId : undefined;
      if (winnerTraceId) {
        event.preventDefault();
        selectWinner(winnerTraceId);
      }
    };
    window.addEventListener('keydown', handleWinnerShortcut);
    return () => window.removeEventListener('keydown', handleWinnerShortcut);
  }, [baselineTraceId, candidateTraceId, selectWinner]);

  const traceDetails = (
    <section
      aria-label={intl.formatMessage({
        defaultMessage: 'Pairwise trace details',
        description: 'Accessible label for the side-by-side trace details in pairwise review',
      })}
      css={{ display: 'flex', flex: 1, minWidth: 0, minHeight: 0, overflow: 'hidden' }}
    >
      <div
        data-pairwise-trace-side="positive"
        css={{ display: 'flex', flex: 1, minWidth: 0, overflow: 'hidden', position: 'relative' }}
      >
        <ModelTraceExplorerUpdateTraceContextProvider
          modelTraceInfo={baselineTrace.info}
          onEditTags={isV3ModelTraceInfo(baselineTrace.info) ? onEditTags : undefined}
        >
          <ModelTraceExplorer
            modelTrace={baselineTrace}
            showSpanTree={showBaselineSpanTree}
            onShowSpanTreeChange={setShowBaselineSpanTree}
            hideAssessments
          />
        </ModelTraceExplorerUpdateTraceContextProvider>
      </div>
      <div
        data-pairwise-trace-side="negative"
        css={{
          display: 'flex',
          flex: 1,
          minWidth: 0,
          overflow: 'hidden',
          position: 'relative',
          borderLeft: `1px solid ${theme.colors.border}`,
        }}
      >
        <ModelTraceExplorerUpdateTraceContextProvider
          modelTraceInfo={candidateTrace.info}
          onEditTags={isV3ModelTraceInfo(candidateTrace.info) ? onEditTags : undefined}
        >
          <ModelTraceExplorer
            modelTrace={candidateTrace}
            showSpanTree={showCandidateSpanTree}
            onShowSpanTreeChange={setShowCandidateSpanTree}
            hideAssessments
          />
        </ModelTraceExplorerUpdateTraceContextProvider>
      </div>
    </section>
  );

  const reviewBanner = (
    <section
      aria-label={intl.formatMessage({
        defaultMessage: 'Pairwise review question',
        description: 'Accessible label for the pairwise review question banner',
      })}
      css={{
        flexShrink: 0,
        padding: theme.spacing.md,
        borderBottom: `1px solid ${theme.colors.border}`,
      }}
    >
      <Card componentId="mlflow.traces-v4.pairwise-review.card" disableHover tabIndex={-1}>
        <div css={{ display: 'flex', minWidth: 0, flexDirection: 'column' }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <CheckboxIcon css={{ flexShrink: 0 }} />
            <Typography.Text bold>
              {intl.formatMessage({
                defaultMessage: 'Which run is better?',
                description: 'Question shown while reviewing two traces side by side',
              })}
            </Typography.Text>
            {isSavingPairwiseReview && (
              <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <Spinner size="small" />
                <Typography.Text size="sm" color="secondary">
                  {intl.formatMessage({
                    defaultMessage: 'Saving…',
                    description: 'Status shown while a pairwise trace review is being saved',
                  })}
                </Typography.Text>
              </span>
            )}
          </div>
          <div css={{ width: '64ch', maxWidth: '100%', marginTop: theme.spacing.sm }}>
            <SegmentedControlGroup
              componentId="mlflow.traces-v4.pairwise-review.question"
              name="mlflow.traces-v4.pairwise-review.question"
              value={questionValue ?? undefined}
              disabled={!baselineTraceInfo || !candidateTraceInfo || isReviewLocked}
              onChange={(event) => {
                const nextSelectedTraceId = event.target.value ? baselineTraceId : candidateTraceId;
                selectWinner(nextSelectedTraceId);
              }}
              style={{ width: '100%' }}
              spaced={false}
            >
              <SegmentedControlButton
                value
                data-pairwise-review-choice="positive"
                style={{ width: '50%', minWidth: 0, justifyContent: 'center' }}
              >
                <span
                  css={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: theme.spacing.xs,
                    minWidth: 0,
                    maxWidth: '100%',
                  }}
                >
                  <RunColorCircle color={theme.colors.textSecondary} />
                  <span
                    css={{
                      display: 'block',
                      minWidth: 0,
                      maxWidth: '40ch',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {baselineLabel}
                  </span>
                  <KeyboardShortcutKey>1</KeyboardShortcutKey>
                </span>
              </SegmentedControlButton>
              <SegmentedControlButton
                value={false}
                data-pairwise-review-choice="negative"
                style={{ width: '50%', minWidth: 0, justifyContent: 'center' }}
              >
                <span
                  css={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: theme.spacing.xs,
                    minWidth: 0,
                    maxWidth: '100%',
                  }}
                >
                  <RunColorCircle color={theme.colors.actionLinkDefault} />
                  <span
                    css={{
                      display: 'block',
                      minWidth: 0,
                      maxWidth: '40ch',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {candidateLabel}
                  </span>
                  <KeyboardShortcutKey>2</KeyboardShortcutKey>
                </span>
              </SegmentedControlButton>
            </SegmentedControlGroup>
          </div>
          <div css={{ width: '64ch', maxWidth: '100%', paddingTop: theme.spacing.sm }}>
            <Input
              id="mlflow-traces-v4-pairwise-review-comment"
              componentId="mlflow.traces-v4.pairwise-review.rationale"
              aria-label={commentLabel}
              value={rationale}
              disabled={isReviewLocked}
              placeholder={intl.formatMessage({
                defaultMessage: 'Add a comment (optional)',
                description: 'Placeholder for an optional comment in pairwise trace review',
              })}
              onChange={(event) => setRationale(event.target.value)}
              onBlur={() => {
                if (selectedTraceId && rationale !== lastSavedRationale && !isReviewLocked) {
                  void persistReview(selectedTraceId, rationale).catch(() => {
                    setRationale(lastSavedRationale);
                  });
                }
              }}
            />
          </div>
        </div>
      </Card>
    </section>
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minWidth: 0,
        minHeight: 0,
        '&:has([data-pairwise-review-choice="positive"]:hover) [data-pairwise-trace-side="positive"], &:has([data-pairwise-review-choice="negative"]:hover) [data-pairwise-trace-side="negative"]':
          {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
          },
        '&:has([data-pairwise-review-choice="positive"]:hover) [data-pairwise-trace-side="positive"] [data-radix-scroll-area-viewport]:has([role="tablist"]), &:has([data-pairwise-review-choice="negative"]:hover) [data-pairwise-trace-side="negative"] [data-radix-scroll-area-viewport]:has([role="tablist"])':
          { background: 'none' },
      }}
    >
      {reviewBanner}
      {traceDetails}
    </div>
  );
};

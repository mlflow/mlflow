import {
  ApplyDesignSystemContextOverrides,
  GenericSkeleton,
  SpeechBubbleIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  DangerIcon,
} from '@databricks/design-system';
import { useEffect, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import {
  createTraceV4LongIdentifier,
  isV3ModelTraceInfo,
  ModelTraceExplorer,
  ModelTraceExplorerDrawer,
  ModelTraceExplorerPreferencesProvider,
  ModelTraceExplorerRunJudgesContextProvider,
  ModelTraceExplorerUpdateTraceContextProvider,
  shouldEnableModelTraceComparisonV2,
  useUnifiedTraceTagsModal,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { useQueryClient } from '@databricks/web-shared/query-client';
import {
  convertTraceInfoV3ToRunEvalEntry,
  createTraceLocationForExperiment,
  doesTraceSupportV4API,
  getTraceV4QueryKey,
  invalidateMlflowSearchTracesCache,
  useSearchMlflowTraces,
} from '@databricks/web-shared/genai-traces-table';
import { useRegisterAssistantContext } from '@mlflow/mlflow/src/assistant';
import { getChatSessionsFilter } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-chat-sessions/utils';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { EvaluationsReviewDetailsHeader } from '@databricks/web-shared/genai-traces-table/components/EvaluationsReviewDetails';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useGetTrace } from '../hooks/useGetTrace';
import type { TracesV4TraceActions } from '../hooks/useTracesV4TraceActions';
import {
  BASELINE_TRACE_ID_PARAM,
  CANDIDATE_TRACE_ID_PARAM,
  PAIRWISE_REVIEW_MODE_PARAM,
  TRACE_ID_PARAM,
} from '../hooks/useTracesV4UrlState';
import { TracesV4PairwiseReviewContent } from './TracesV4PairwiseReviewContent';
import { TracesV4SessionDrawerBody } from './TracesV4SessionDrawerContent';

/** A session opened in the drawer: its conversation, or one of its traces with a way back. */
export interface TracesV4TraceDrawerSession {
  sessionId: string;
  viewMode: 'trace' | 'session';
  /** Shows one of the session's traces (by its URL id) in the drawer. */
  onViewTrace: (traceId: string) => void;
  /** Returns from a trace to the session's conversation. */
  onBackToSession: () => void;
  /** Moves to the previous or next session on the page; omitted when there is none. */
  onPreviousSession?: () => void;
  onNextSession?: () => void;
}

/** A run of the opened evaluation-comparison row; `traceId` is absent when the run has no trace for it. */
export interface TracesV4DrawerComparisonRunOption {
  runUuid: string;
  runName: string;
  traceId?: string;
}

/** Moves between matched comparison rows while keeping both compared traces in sync. */
export interface TracesV4DrawerComparisonNavigation {
  onPrevious?: () => void;
  onNext?: () => void;
  currentIndex: number;
  totalCount: number;
}

export interface TracesV4TraceDrawerProps {
  /** Long id (`trace:/<catalog.schema>/<id>`) of the open trace, or undefined when the drawer is closed. */
  traceId?: string;
  onClose: () => void;
  experimentId: string;
  /** The current page's traces — next/back navigates within this list only. */
  traces: ModelTraceInfoV3[];
  /** Writes the newly-selected trace's id to the URL (as a long id — see {@link idOf}). */
  onSelectTrace: (traceId: string) => void;
  /** Optional in-explorer run-judges config (enables "Run judge" from within a span's assessments). */
  runJudgeConfiguration?: NonNullable<TracesV4TraceActions['runJudges']>['runJudgeConfiguration'];
  /** Opens the drawer on a session instead of a single trace. Omit for trace-only drawers. */
  session?: TracesV4TraceDrawerSession;
  /** Runs represented by the opened evaluation-comparison row; enables picking each side's run. */
  comparisonRunOptions?: TracesV4DrawerComparisonRunOption[];
  /** Previous/next matched rows while comparing; without it, previous/next are off in comparison mode. */
  comparisonNavigation?: TracesV4DrawerComparisonNavigation;
}

const SESSION_ID_PREVIEW_LENGTH = 8;

/** The id the URL uses for a trace: the V4 long id for UC-backed traces, the bare id otherwise. */
const idOf = (trace: ModelTraceInfoV3): string =>
  doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id;

/**
 * Detail drawer for the v4 Traces tab. Renders the shared `ModelTraceExplorerDrawer` (replacing the
 * bare `TraceModal`) so v4 gets next/back navigation, Share, "Add to dataset", "Flag for review",
 * and "Add to labeling session" — reusing the shared drawer with zero v3/shared-package edits.
 *
 * Next/Back navigate within the current page only (v4 uses server-side numbered pagination); users
 * cross pages via the pagination bar. A deep-linked trace not on the page still loads its body via
 * `useGetTrace`, but with prev/next disabled and the trace-scoped header actions hidden (they need
 * the row's `ModelTraceInfoV3`, which only on-page rows carry).
 */
export const TracesV4TraceDrawer = ({
  traceId,
  onClose,
  experimentId,
  traces,
  onSelectTrace,
  runJudgeConfiguration,
  session,
  comparisonRunOptions,
  comparisonNavigation,
}: TracesV4TraceDrawerProps) => {
  const { theme } = useDesignSystemTheme();
  // A session's turns in order (URL ids), reported by `SessionTurnsLoader` while one of its traces is open.
  const [sessionTurnIds, setSessionTurnIds] = useState<string[]>();

  // Depends only on props (`traces`, `traceId`), so it's safe to compute before the hooks below and
  // feed the clicked row's timing into `useGetTrace` for time-hint derivation. `undefined` for a
  // deep-link whose row isn't on the current page.
  const currentIndex = traces.findIndex((trace) => idOf(trace) === traceId);
  const currentInfo = currentIndex >= 0 ? traces[currentIndex] : undefined;

  // `useGetTrace` parses the `trace:/…` long id itself, so it also serves deep-links where the row
  // isn't on the current page. Passing the row's info lets it derive BatchGetTraces time hints.
  const { data: traceData, isLoading, error } = useGetTrace(traceId ?? '', undefined, currentInfo);

  // With the new eval runs UI, `baselineTraceId` + `candidateTraceId` in the URL open the drawer as a
  // comparison of the two traces; the open trace is one side and its counterpart is loaded alongside it.
  const [searchParams, setSearchParams] = useSearchParams();
  const baselineTraceId = searchParams.get(BASELINE_TRACE_ID_PARAM) ?? undefined;
  const candidateTraceId = searchParams.get(CANDIDATE_TRACE_ID_PARAM) ?? undefined;
  const isComparisonMode = Boolean(
    shouldEnableModelTraceComparisonV2() &&
    baselineTraceId &&
    candidateTraceId &&
    (traceId === baselineTraceId || traceId === candidateTraceId),
  );
  const comparisonRole = isComparisonMode && traceId === baselineTraceId ? 'baseline' : 'candidate';
  // Pairwise review shows both compared traces side by side; it only exists inside comparison mode.
  const isPairwiseReviewMode = isComparisonMode && searchParams.get(PAIRWISE_REVIEW_MODE_PARAM) === 'true';
  const counterpartTraceId = isComparisonMode
    ? comparisonRole === 'baseline'
      ? candidateTraceId
      : baselineTraceId
    : undefined;
  const counterpartInfo = traces.find((trace) => idOf(trace) === counterpartTraceId);
  // Disabled (no request) while `counterpartTraceId` is empty, i.e. outside comparison mode.
  const {
    data: counterpartTraceData,
    isLoading: isCounterpartLoading,
    error: counterpartError,
  } = useGetTrace(counterpartTraceId ?? '', undefined, counterpartInfo);

  // "Edit" (key/value tags) reuses the shared unified tag modal. On save, refresh both the list rows
  // and the open drawer body so the new tags show without a manual reload.
  const queryClient = useQueryClient();
  const { showTagAssignmentModal, TagAssignmentModal } = useUnifiedTraceTagsModal({
    componentIdPrefix: 'mlflow.traces-v4',
    onSuccess: () => {
      invalidateMlflowSearchTracesCache({ queryClient }); // list rows
      const editedTraceId = traceData?.info && isV3ModelTraceInfo(traceData.info) ? traceData.info.trace_id : undefined;
      // The drawer header's tags refresh instantly via the shared mutation's optimistic
      // trace-info-cache write, so we deliberately do NOT invalidate `FETCH_TRACE_INFO_QUERY_KEY`
      // here — that forced a slow refetch on the critical path and, because the tag-write backend is
      // eventually consistent, risked reading back stale tags and reverting the correct value.
      if (editedTraceId) {
        // `useGetTrace` layers a query over an inner `ensureQueryData` cache that ignores staleness;
        // remove that inner entry so a re-open / navigation refetches fresh tags.
        queryClient.removeQueries({ queryKey: getTraceV4QueryKey(editedTraceId) });
      }
      // Matches `useGetTrace`'s query key so the drawer's own view reflects the edit.
      queryClient.invalidateQueries({ queryKey: ['getTrace'] });
    },
  });

  if (session?.viewMode === 'session') {
    return (
      <ModelTraceExplorerDrawer
        handleClose={onClose}
        selectPreviousEval={session.onPreviousSession ?? (() => {})}
        selectNextEval={session.onNextSession ?? (() => {})}
        isPreviousAvailable={Boolean(session.onPreviousSession)}
        isNextAvailable={Boolean(session.onNextSession)}
        renderModalTitle={() => <Typography.Text bold>{session.sessionId}</Typography.Text>}
        headerTitle={<SessionDrawerTitle sessionId={session.sessionId} />}
        isLoading={false}
        experimentId={experimentId}
      >
        <TracesV4SessionDrawerBody
          experimentId={experimentId}
          sessionId={session.sessionId}
          selectedTraceId={traceId}
          onViewTrace={(traceInfo) => session.onViewTrace(idOf(traceInfo))}
        />
      </ModelTraceExplorerDrawer>
    );
  }

  // Only mount once a trace is selected.
  if (!traceId) {
    return null;
  }

  // A trace opened from a session moves between the session's turns, as on the session page.
  // Comparison mode moves between matched comparison rows when the caller provides them.
  const turnIndex = session && sessionTurnIds ? sessionTurnIds.indexOf(traceId) : -1;
  const isPreviousAvailable = isComparisonMode
    ? Boolean(comparisonNavigation?.onPrevious)
    : session
      ? turnIndex > 0
      : currentIndex > 0;
  const isNextAvailable = isComparisonMode
    ? Boolean(comparisonNavigation?.onNext)
    : session
      ? turnIndex >= 0 && sessionTurnIds !== undefined && turnIndex < sessionTurnIds.length - 1
      : currentIndex >= 0 && currentIndex < traces.length - 1;
  const selectPreviousEval = () => {
    if (!isPreviousAvailable) {
      return;
    }
    if (isComparisonMode) {
      comparisonNavigation?.onPrevious?.();
      return;
    }
    if (session && sessionTurnIds) {
      session.onViewTrace(sessionTurnIds[turnIndex - 1]);
    } else {
      onSelectTrace(idOf(traces[currentIndex - 1]));
    }
  };
  const selectNextEval = () => {
    if (!isNextAvailable) {
      return;
    }
    if (isComparisonMode) {
      comparisonNavigation?.onNext?.();
      return;
    }
    if (session && sessionTurnIds) {
      session.onViewTrace(sessionTurnIds[turnIndex + 1]);
    } else {
      onSelectTrace(idOf(traces[currentIndex + 1]));
    }
  };

  const handleSwapComparisonOrder = () => {
    if (!baselineTraceId || !candidateTraceId) {
      return;
    }
    setSearchParams((params) => {
      params.set(BASELINE_TRACE_ID_PARAM, candidateTraceId);
      params.set(CANDIDATE_TRACE_ID_PARAM, baselineTraceId);
      params.set(TRACE_ID_PARAM, baselineTraceId);
      params.delete('spanId');
      return params;
    });
  };

  const handleSelectBaselineTrace = (selectedTraceId: string) => {
    if (!candidateTraceId || selectedTraceId === candidateTraceId) {
      return;
    }
    setSearchParams((params) => {
      params.set(BASELINE_TRACE_ID_PARAM, selectedTraceId);
      params.set(TRACE_ID_PARAM, candidateTraceId);
      params.delete('spanId');
      return params;
    });
  };

  const handleSelectCandidateTrace = (selectedTraceId: string) => {
    if (!baselineTraceId || selectedTraceId === baselineTraceId) {
      return;
    }
    setSearchParams((params) => {
      params.set(CANDIDATE_TRACE_ID_PARAM, selectedTraceId);
      params.set(TRACE_ID_PARAM, selectedTraceId);
      params.delete('spanId');
      return params;
    });
  };

  const getStandaloneTraceUrl = (selectedTraceId: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set(TRACE_ID_PARAM, selectedTraceId);
    url.searchParams.delete(BASELINE_TRACE_ID_PARAM);
    url.searchParams.delete(CANDIDATE_TRACE_ID_PARAM);
    url.searchParams.delete(PAIRWISE_REVIEW_MODE_PARAM);
    url.searchParams.delete('spanId');
    return url.toString();
  };

  const handleReviewModeChange = (isReviewMode: boolean) => {
    setSearchParams((params) => {
      if (isReviewMode) {
        params.set(PAIRWISE_REVIEW_MODE_PARAM, 'true');
      } else {
        params.delete(PAIRWISE_REVIEW_MODE_PARAM);
      }
      return params;
    });
  };

  // The drawer renders its own skeleton while `isLoading`, so this only covers the settled states.
  const renderBody = () => {
    if (error) {
      return <TraceLoadError traceId={traceId} />;
    }
    if (isComparisonMode && counterpartError) {
      return <TraceLoadError traceId={counterpartTraceId ?? ''} />;
    }
    if (isPairwiseReviewMode && traceData && counterpartTraceData && baselineTraceId && candidateTraceId) {
      const baselineTrace = comparisonRole === 'baseline' ? traceData : counterpartTraceData;
      const candidateTrace = comparisonRole === 'candidate' ? traceData : counterpartTraceData;
      return (
        <div
          css={{
            height: '100%',
            display: 'flex',
            minHeight: 0,
            marginLeft: -theme.spacing.lg,
            marginRight: -theme.spacing.lg,
          }}
        >
          <ModelTraceExplorerPreferencesProvider>
            <TracesV4PairwiseReviewContent
              key={`${baselineTraceId}:${candidateTraceId}`}
              baselineTrace={baselineTrace}
              baselineTraceId={baselineTraceId}
              baselineRunName={comparisonRunOptions?.find((option) => option.traceId === baselineTraceId)?.runName}
              candidateTrace={candidateTrace}
              candidateTraceId={candidateTraceId}
              candidateRunName={comparisonRunOptions?.find((option) => option.traceId === candidateTraceId)?.runName}
              onEditTags={showTagAssignmentModal}
            />
          </ModelTraceExplorerPreferencesProvider>
        </div>
      );
    }
    if (traceData && (!isComparisonMode || counterpartTraceData)) {
      const explorer = (
        <div css={{ height: '100%', marginLeft: -theme.spacing.lg, marginRight: -theme.spacing.lg }}>
          <ModelTraceExplorerPreferencesProvider>
            <ModelTraceExplorerUpdateTraceContextProvider
              modelTraceInfo={traceData.info}
              // Surfaces the Edit/Add-tags button next to the header's Tags field. `showTagAssignmentModal`
              // takes `ModelTrace['info']`, so the header's own trace info flows straight to the modal —
              // the modal edits exactly the trace shown. Gate on the shown trace being editable V3 info
              // (has a trace identity); a resolved-but-unidentifiable trace (no location) offers no edit.
              onEditTags={isV3ModelTraceInfo(traceData.info) ? showTagAssignmentModal : undefined}
            >
              {/* A session's trace opens with its assessments showing, as on the session page. */}
              <ModelTraceExplorer
                modelTrace={traceData}
                {...(session ? { collapseAssessmentPane: 'force-open' as const } : {})}
                comparisonModelTrace={isComparisonMode ? counterpartTraceData : undefined}
                comparisonRole={isComparisonMode ? comparisonRole : undefined}
              />
            </ModelTraceExplorerUpdateTraceContextProvider>
          </ModelTraceExplorerPreferencesProvider>
        </div>
      );
      if (!session) {
        return explorer;
      }
      return (
        <div css={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <SessionTurnsLoader
            experimentId={experimentId}
            sessionId={session.sessionId}
            traceId={traceId}
            onTurnsLoaded={setSessionTurnIds}
          />
          <SessionTraceBanner sessionId={session.sessionId} onBackToSession={session.onBackToSession} />
          <div css={{ flex: 1, minHeight: 0 }}>{explorer}</div>
        </div>
      );
    }
    return <TraceNoData traceId={isComparisonMode ? (counterpartTraceId ?? '') : traceId} />;
  };

  const body = runJudgeConfiguration ? (
    <ModelTraceExplorerRunJudgesContextProvider {...runJudgeConfiguration}>
      {renderBody()}
    </ModelTraceExplorerRunJudgesContextProvider>
  ) : (
    renderBody()
  );

  // Header title mirrors v3: the trace's input preview (falling back to the id) rather than the raw
  // id. Prefer the row's info (present for on-page rows), else the fetched trace's V3 info (covers
  // deep-links not on the current page); while that fetch is in flight, show a skeleton.
  const fetchedInfo = traceData?.info;
  const titleInfo = currentInfo ?? (fetchedInfo && isV3ModelTraceInfo(fetchedInfo) ? fetchedInfo : undefined);
  const renderModalTitle = () => {
    if (titleInfo) {
      return <EvaluationsReviewDetailsHeader evaluationResult={convertTraceInfoV3ToRunEvalEntry(titleInfo)} />;
    }
    if (isLoading) {
      return <GenericSkeleton css={{ width: 200, height: theme.general.heightBase }} />;
    }
    // No trace info resolved (e.g. load error) — fall back to the raw id so the header isn't empty.
    return <Typography.Text bold>{traceId}</Typography.Text>;
  };

  // In pairwise review, the position among the matched comparison rows sits between previous and next.
  const navigationLabel =
    isPairwiseReviewMode &&
    comparisonNavigation &&
    comparisonNavigation.currentIndex >= 0 &&
    comparisonNavigation.totalCount > 0 ? (
      <Tag componentId="mlflow.traces-v4.trace-context.comparison-position" color="default" css={{ margin: 0 }}>
        <Typography.Text size="sm" color="secondary" css={{ whiteSpace: 'nowrap' }}>
          <FormattedMessage
            defaultMessage="Trace {traceNumber} of {traceCount}"
            description="Current trace comparison position shown between pairwise review navigation buttons"
            values={{
              traceNumber: comparisonNavigation.currentIndex + 1,
              traceCount: comparisonNavigation.totalCount,
            }}
          />
        </Typography.Text>
      </Tag>
    ) : undefined;

  return (
    <>
      <ModelTraceExplorerDrawer
        handleClose={onClose}
        navigationLabel={navigationLabel}
        selectPreviousEval={selectPreviousEval}
        selectNextEval={selectNextEval}
        isPreviousAvailable={isPreviousAvailable}
        isNextAvailable={isNextAvailable}
        renderModalTitle={renderModalTitle}
        isLoading={isLoading || (isComparisonMode && isCounterpartLoading)}
        experimentId={experimentId}
        traceInfo={currentInfo}
        traceComparison={
          isComparisonMode && baselineTraceId && candidateTraceId
            ? {
                baselineTraceId,
                candidateTraceId,
                baselineTraceUrl: getStandaloneTraceUrl(baselineTraceId),
                candidateTraceUrl: getStandaloneTraceUrl(candidateTraceId),
                onSwap: handleSwapComparisonOrder,
                baselineTraceInfo:
                  comparisonRole === 'baseline'
                    ? isV3ModelTraceInfo(traceData?.info)
                      ? traceData.info
                      : undefined
                    : isV3ModelTraceInfo(counterpartTraceData?.info)
                      ? counterpartTraceData.info
                      : undefined,
                candidateTraceInfo:
                  comparisonRole === 'candidate'
                    ? isV3ModelTraceInfo(traceData?.info)
                      ? traceData.info
                      : undefined
                    : isV3ModelTraceInfo(counterpartTraceData?.info)
                      ? counterpartTraceData.info
                      : undefined,
                isReviewMode: isPairwiseReviewMode,
                onReviewModeChange: handleReviewModeChange,
                runOptions: comparisonRunOptions,
                onSelectBaselineTrace: comparisonRunOptions ? handleSelectBaselineTrace : undefined,
                onSelectCandidateTrace: comparisonRunOptions ? handleSelectCandidateTrace : undefined,
              }
            : undefined
        }
      >
        {body}
      </ModelTraceExplorerDrawer>
      {/* The modal is a sibling of the drawer, so it inherits the ambient zIndexBase and would render
          behind the drawer panel (which sits at zIndexBase + 2). Elevate it the same way the drawer
          elevates its own body children so the modal — and its backdrop — sit above the drawer. */}
      <ApplyDesignSystemContextOverrides zIndexBase={2 * theme.options.zIndexBase}>
        {TagAssignmentModal}
      </ApplyDesignSystemContextOverrides>
    </>
  );
};

/** Drawer title for a session: "Session" and its id, which the full id shows on hover. */
const SessionDrawerTitle = ({ sessionId }: { sessionId: string }) => (
  <>
    <Typography.Text css={{ whiteSpace: 'nowrap' }}>
      <FormattedMessage defaultMessage="Session" description="Title for the trace drawer when it shows a session" />
    </Typography.Text>
    <Tooltip componentId="mlflow.traces-v4.session-drawer.session-id-tooltip" content={sessionId}>
      <Tag componentId="mlflow.traces-v4.session-drawer.session-id" color="default" css={{ margin: 0, maxWidth: 160 }}>
        <Typography.Text size="sm" color="secondary" ellipsis css={{ fontFamily: 'monospace' }}>
          {sessionId}
        </Typography.Text>
      </Tag>
    </Tooltip>
  </>
);

/**
 * Loads the turns of the session a trace was opened from, so the drawer's previous/next buttons move between
 * them, and tells the assistant which session and trace are open. Renders nothing.
 */
const SessionTurnsLoader = ({
  experimentId,
  sessionId,
  traceId,
  onTurnsLoaded,
}: {
  experimentId: string;
  sessionId: string;
  traceId: string;
  onTurnsLoaded: (turnIds: string[]) => void;
}) => {
  const locations = useMemo(() => [createTraceLocationForExperiment(experimentId)], [experimentId]);
  const filters = useMemo(() => getChatSessionsFilter({ sessionId }), [sessionId]);
  const { data: traceInfos } = useSearchMlflowTraces({ locations, filters });
  useEffect(() => {
    if (traceInfos) {
      onTurnsLoaded(
        [...traceInfos].sort((left, right) => Date.parse(left.request_time) - Date.parse(right.request_time)).map(idOf),
      );
    }
  }, [traceInfos, onTurnsLoaded]);
  useRegisterAssistantContext('sessionId', sessionId);
  useRegisterAssistantContext('traceId', traceId);
  return null;
};

/** Context shown above a trace opened from a session's conversation, with a way back to it. */
const SessionTraceBanner = ({ sessionId, onBackToSession }: { sessionId: string; onBackToSession: () => void }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        minWidth: 0,
        paddingBottom: theme.spacing.sm,
      }}
    >
      <SpeechBubbleIcon css={{ flexShrink: 0 }} />
      <Typography.Text size="sm" color="secondary" css={{ whiteSpace: 'nowrap' }}>
        <FormattedMessage
          defaultMessage="Trace from session"
          description="Context shown above a trace opened from a session conversation"
        />
      </Typography.Text>
      <Tooltip componentId="mlflow.traces-v4.trace-context.session-id-tooltip" content={sessionId}>
        <Tag componentId="mlflow.traces-v4.trace-context.session-id" color="default" css={{ margin: 0, maxWidth: 96 }}>
          <Typography.Text size="sm" color="secondary" css={{ fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
            {sessionId.slice(0, SESSION_ID_PREVIEW_LENGTH)}
          </Typography.Text>
        </Tag>
      </Tooltip>
      <Typography.Text size="sm" color="secondary">
        ·
      </Typography.Text>
      <Typography.Link
        componentId="mlflow.traces-v4.trace-context.back-to-session"
        href="#"
        css={{ fontSize: theme.typography.fontSizeSm, whiteSpace: 'nowrap' }}
        onClick={(event) => {
          event.preventDefault();
          onBackToSession();
        }}
      >
        <FormattedMessage
          defaultMessage="Back to session"
          description="Action to return from a full trace to its session conversation"
        />
      </Typography.Link>
    </div>
  );
};

/** Centered flex-column container shared by the drawer's error / no-data states. */
const DrawerCenteredState = ({ children }: { children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: 300,
        gap: theme.spacing.md,
      }}
    >
      {children}
    </div>
  );
};

/** Error state shown in the drawer body when the trace fails to load. */
const TraceLoadError = ({ traceId }: { traceId: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <DrawerCenteredState>
      <DangerIcon css={{ fontSize: 48, color: theme.colors.actionDangerPrimaryBackgroundDefault }} />
      <Typography.Title level={4}>
        <FormattedMessage
          defaultMessage="Failed to load trace"
          description="Title shown in the V4 trace drawer when the trace fails to load"
        />
      </Typography.Title>
      <Typography.Text>
        <FormattedMessage
          defaultMessage="Unable to fetch trace data for trace ID: {traceId}"
          description="Body shown in the V4 trace drawer when the trace fails to load"
          values={{ traceId }}
        />
      </Typography.Text>
    </DrawerCenteredState>
  );
};

/** Empty state shown in the drawer body when the trace resolved but has no data. */
const TraceNoData = ({ traceId }: { traceId: string }) => (
  <DrawerCenteredState>
    <Typography.Title level={4}>
      <FormattedMessage
        defaultMessage="No trace data available"
        description="Title shown in the V4 trace drawer when a trace has no data"
      />
    </Typography.Title>
    <Typography.Text>
      <FormattedMessage
        defaultMessage="No data found for trace ID: {traceId}"
        description="Body shown in the V4 trace drawer when a trace has no data"
        values={{ traceId }}
      />
    </Typography.Text>
  </DrawerCenteredState>
);

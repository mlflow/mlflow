import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
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
import { FormattedMessage } from 'react-intl';
import {
  createTraceV4LongIdentifier,
  type ModelTraceSearchLocation,
  isV3ModelTraceInfo,
  ModelTraceExplorer,
  ModelTraceExplorerDrawer,
  ModelTraceExplorerPreferencesProvider,
  ModelTraceExplorerRunJudgesContextProvider,
  ModelTraceExplorerUpdateTraceContextProvider,
  useUnifiedTraceTagsModal,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { useQueryClient } from '@databricks/web-shared/query-client';
import {
  convertTraceInfoV3ToRunEvalEntry,
  doesTraceSupportV4API,
  GenAITracesTableContext,
  getTraceV4QueryKey,
  invalidateMlflowSearchTracesCache,
  useGetTraces,
  useSearchMlflowTraces,
} from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { EvaluationsReviewDetailsHeader } from '@databricks/web-shared/genai-traces-table/components/EvaluationsReviewDetails';
import { useGetTrace } from '../hooks/useGetTrace';
import type { TracesV4TraceActions } from '../hooks/useTracesV4TraceActions';
import { TracesV4SessionDrawerContent } from './TracesV4SessionDrawerContent';
import { useExperimentSingleChatMetrics } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-chat-sessions/single-chat-view/useExperimentSingleChatMetrics';
import { getChatSessionsFilter } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-chat-sessions/utils';
import {
  TRACE_DRAWER_SESSION_ID_QUERY_PARAM,
  TRACE_DRAWER_VIEW_MODE_QUERY_PARAM,
} from '@mlflow/mlflow/src/experiment-tracking/constants';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';

export interface TracesV4DrawerViewModeControl {
  value: 'trace' | 'session';
  onChange: (viewMode: 'trace' | 'session') => void;
}

export interface TracesV4DrawerSession {
  sessionId: string;
  traceId: string;
  traceLocation?: ModelTraceSearchLocation;
}

export interface TracesV4DrawerSessionNavigation {
  onPrevious?: () => void;
  onNext?: () => void;
}

export interface TracesV4TraceDrawerProps {
  /** Long id (`trace:/<catalog.schema>/<id>`) of the open trace, or undefined when the drawer is closed. */
  traceId?: string;
  onClose: () => void;
  experimentId: string;
  /** The current page's traces — next/back navigates within this list only. */
  traces: ModelTraceInfoV3[];
  /** Writes the newly-selected trace's id to the URL (as a long id — see {@link idOf}). */
  onSelectTrace: (traceId: string, traceInfo?: ModelTraceInfoV3) => void;
  /** Optional in-explorer run-judges config (enables "Run judge" from within a span's assessments). */
  runJudgeConfiguration?: NonNullable<TracesV4TraceActions['runJudges']>['runJudgeConfiguration'];
  viewModeControl?: TracesV4DrawerViewModeControl;
  session?: TracesV4DrawerSession;
  sessionNavigation?: TracesV4DrawerSessionNavigation;
}

/** The id the URL uses for a trace: the V4 long id for UC-backed traces, the bare id otherwise. */
const idOf = (trace: ModelTraceInfoV3): string =>
  doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id;

const SESSION_ID_PREVIEW_LENGTH = 8;

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
  viewModeControl,
  session,
  sessionNavigation,
}: TracesV4TraceDrawerProps) => {
  const { theme } = useDesignSystemTheme();
  const drawerViewMode = viewModeControl?.value ?? 'trace';
  const sessionId = session?.sessionId;
  const isSessionView = drawerViewMode === 'session' && Boolean(session);
  const [sessionTraceNavigationSessionId, setSessionTraceNavigationSessionId] = useState<string>();

  // Depends only on props (`traces`, `traceId`), so it's safe to compute before the hooks below and
  // feed the clicked row's timing into `useGetTrace` for time-hint derivation. `undefined` for a
  // deep-link whose row isn't on the current page.
  const currentIndex = traces.findIndex((trace) => idOf(trace) === traceId);
  const currentInfo = currentIndex >= 0 ? traces[currentIndex] : undefined;

  // `useGetTrace` parses the `trace:/…` long id itself, so it also serves deep-links where the row
  // isn't on the current page. Passing the row's info lets it derive BatchGetTraces time hints.
  const { data: traceData, isLoading, error } = useGetTrace(traceId ?? '', undefined, currentInfo);
  const sessionFilters = useMemo(() => (sessionId ? getChatSessionsFilter({ sessionId }) : []), [sessionId]);
  const fetchedTraceInfo = traceData?.info;
  const fetchedTraceLocation = isV3ModelTraceInfo(fetchedTraceInfo) ? fetchedTraceInfo.trace_location : undefined;
  const sessionTraceLocations = useMemo<ModelTraceSearchLocation[]>(() => {
    const traceLocation = session?.traceLocation ?? fetchedTraceLocation;
    return traceLocation && traceLocation.type !== 'INFERENCE_TABLE' ? [traceLocation] : [];
  }, [fetchedTraceLocation, session?.traceLocation]);
  const {
    data: sessionTraceInfos,
    isLoading: isLoadingSessionTraceInfos,
    isFetching: isFetchingSessionTraceInfos,
    error: sessionTraceInfosError,
    refetchMlflowTraces: refetchSessionTraceInfos,
  } = useSearchMlflowTraces({
    locations: sessionTraceLocations,
    filters: sessionFilters,
    disabled: !isSessionView || sessionTraceLocations.length === 0,
    fetchAllPages: true,
    enablePagination: false,
  });
  const orderedSessionTraceInfos = useMemo(
    () =>
      [...(sessionTraceInfos ?? [])].sort(
        (left, right) => Date.parse(left.request_time) - Date.parse(right.request_time),
      ),
    [sessionTraceInfos],
  );
  const selectedSessionTraceId = orderedSessionTraceInfos.find(
    (trace) => trace.trace_id === session?.traceId || idOf(trace) === session?.traceId,
  )?.trace_id;
  const sessionTraceInfosHaveSettled =
    sessionTraceLocations.length === 0 ||
    Boolean(selectedSessionTraceId) ||
    (!isLoadingSessionTraceInfos && !isFetchingSessionTraceInfos);
  const currentSessionTraceInfos = useMemo(
    () => (sessionTraceInfosHaveSettled ? orderedSessionTraceInfos : []),
    [orderedSessionTraceInfos, sessionTraceInfosHaveSettled],
  );
  const loadedSessionTraceInfosRef = useRef<{ sessionId: string; traceInfos: ModelTraceInfoV3[] }>();
  useEffect(() => {
    if (isSessionView && sessionId && sessionTraceInfosHaveSettled) {
      loadedSessionTraceInfosRef.current = { sessionId, traceInfos: currentSessionTraceInfos };
    }
  }, [currentSessionTraceInfos, isSessionView, sessionId, sessionTraceInfosHaveSettled]);
  const loadedSessionTraceInfos =
    sessionId && loadedSessionTraceInfosRef.current?.sessionId === sessionId
      ? loadedSessionTraceInfosRef.current.traceInfos
      : [];
  const sessionMetrics = useExperimentSingleChatMetrics({
    traceInfos: isSessionView ? currentSessionTraceInfos : undefined,
  });
  const { getTrace } = useContext(GenAITracesTableContext);
  const {
    data: sessionTraces,
    isLoading: isLoadingSessionTraceData,
    error: sessionTraceDataError,
    invalidateSingleTraceQuery,
  } = useGetTraces(getTrace, currentSessionTraceInfos);
  const refreshSession = useCallback(() => {
    refetchSessionTraceInfos?.();
    currentSessionTraceInfos.forEach((traceInfo) => invalidateSingleTraceQuery(traceInfo.trace_id));
  }, [currentSessionTraceInfos, invalidateSingleTraceQuery, refetchSessionTraceInfos]);
  const sessionShareUrl = useMemo(() => {
    if (!session) {
      return undefined;
    }
    const searchParams = new URLSearchParams();
    searchParams.set('groupBy', 'session');
    searchParams.set('startTimeLabel', 'ALL');
    searchParams.set('traceId', session.traceId);
    searchParams.set(TRACE_DRAWER_VIEW_MODE_QUERY_PARAM, 'session');
    searchParams.set(TRACE_DRAWER_SESSION_ID_QUERY_PARAM, session.sessionId);
    return `${window.location.origin}${Routes.getExperimentPageTracesTabRoute(experimentId)}?${searchParams}`;
  }, [experimentId, session]);
  const handleDrawerViewModeChange = useCallback(
    (viewMode: 'trace' | 'session') => {
      setSessionTraceNavigationSessionId(undefined);
      viewModeControl?.onChange(viewMode);
    },
    [viewModeControl],
  );

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

  if (!traceId && !(drawerViewMode === 'session' && sessionId)) {
    return null;
  }

  const isPreviousAvailable = currentIndex > 0;
  const isNextAvailable = currentIndex >= 0 && currentIndex < traces.length - 1;
  const sessionTurnIndex = loadedSessionTraceInfos.findIndex((trace) => idOf(trace) === traceId);
  const isSessionTurnTraceView =
    !isSessionView && sessionTraceNavigationSessionId === sessionId && sessionTurnIndex >= 0;
  const selectPreviousEval = () => {
    if (isPreviousAvailable) {
      const traceInfo = traces[currentIndex - 1];
      onSelectTrace(idOf(traceInfo), traceInfo);
    }
  };
  const selectNextEval = () => {
    if (isNextAvailable) {
      const traceInfo = traces[currentIndex + 1];
      onSelectTrace(idOf(traceInfo), traceInfo);
    }
  };
  const selectPreviousSessionTurn = () => {
    if (sessionTurnIndex > 0) {
      const traceInfo = loadedSessionTraceInfos[sessionTurnIndex - 1];
      onSelectTrace(idOf(traceInfo), traceInfo);
    }
  };
  const selectNextSessionTurn = () => {
    if (sessionTurnIndex >= 0 && sessionTurnIndex < loadedSessionTraceInfos.length - 1) {
      const traceInfo = loadedSessionTraceInfos[sessionTurnIndex + 1];
      onSelectTrace(idOf(traceInfo), traceInfo);
    }
  };
  const selectPrevious = isSessionView
    ? (sessionNavigation?.onPrevious ?? (() => {}))
    : isSessionTurnTraceView
      ? selectPreviousSessionTurn
      : selectPreviousEval;
  const selectNext = isSessionView
    ? (sessionNavigation?.onNext ?? (() => {}))
    : isSessionTurnTraceView
      ? selectNextSessionTurn
      : selectNextEval;
  const previousAvailable = isSessionView
    ? Boolean(sessionNavigation?.onPrevious)
    : isSessionTurnTraceView
      ? sessionTurnIndex > 0
      : isPreviousAvailable;
  const nextAvailable = isSessionView
    ? Boolean(sessionNavigation?.onNext)
    : isSessionTurnTraceView
      ? sessionTurnIndex < loadedSessionTraceInfos.length - 1
      : isNextAvailable;
  const sessionIdPreview = sessionId?.slice(0, SESSION_ID_PREVIEW_LENGTH);
  const sessionTraceHeaderBanner =
    isSessionTurnTraceView && sessionId && viewModeControl ? (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, minWidth: 0 }}>
        <SpeechBubbleIcon css={{ flexShrink: 0 }} />
        <Typography.Text size="sm" color="secondary" css={{ whiteSpace: 'nowrap' }}>
          <FormattedMessage defaultMessage="Trace from session" description="Trace drawer session context" />
        </Typography.Text>
        <Tooltip componentId="mlflow.traces-v4.trace-context.session-id-tooltip" content={sessionId}>
          <Tag
            componentId="mlflow.traces-v4.trace-context.session-id"
            color="default"
            css={{ margin: 0, maxWidth: 96 }}
          >
            <Typography.Text size="sm" color="secondary" css={{ fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
              {sessionIdPreview}
            </Typography.Text>
          </Tag>
        </Tooltip>
        <Typography.Text size="sm" color="secondary">
          ·
        </Typography.Text>
        <Typography.Link
          componentId="mlflow.traces-v4.trace-context.back-to-session"
          href="#"
          css={{
            fontSize: theme.typography.fontSizeSm,
            lineHeight: theme.typography.lineHeightSm,
            whiteSpace: 'nowrap',
          }}
          onClick={(event) => {
            event.preventDefault();
            handleDrawerViewModeChange('session');
          }}
        >
          <FormattedMessage defaultMessage="Back to session" description="Return to the session conversation" />
        </Typography.Link>
      </div>
    ) : undefined;
  const sessionTurnNavigationLabel = isSessionTurnTraceView ? (
    <Tag componentId="mlflow.traces-v4.trace-context.turn-position" color="default" css={{ margin: 0 }}>
      <Typography.Text size="sm" color="secondary" css={{ whiteSpace: 'nowrap' }}>
        <FormattedMessage
          defaultMessage="Turn {turnNumber} of {turnCount}"
          description="Current session turn position"
          values={{ turnNumber: sessionTurnIndex + 1, turnCount: loadedSessionTraceInfos.length }}
        />
      </Typography.Text>
    </Tag>
  ) : undefined;

  // The drawer renders its own skeleton while `isLoading`, so this only covers the settled states.
  const renderBody = () => {
    if (error) {
      return <TraceLoadError traceId={traceId ?? ''} />;
    }
    if (traceData) {
      return (
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
              <ModelTraceExplorer modelTrace={traceData} />
            </ModelTraceExplorerUpdateTraceContextProvider>
          </ModelTraceExplorerPreferencesProvider>
        </div>
      );
    }
    return <TraceNoData traceId={traceId ?? ''} />;
  };

  const traceBody = runJudgeConfiguration ? (
    <ModelTraceExplorerRunJudgesContextProvider {...runJudgeConfiguration}>
      {renderBody()}
    </ModelTraceExplorerRunJudgesContextProvider>
  ) : (
    renderBody()
  );
  const sessionBody =
    drawerViewMode === 'session' && sessionId ? (
      <TracesV4SessionDrawerContent
        sessionId={sessionId}
        selectedTraceId={selectedSessionTraceId}
        onRefreshSession={refreshSession}
        traces={sessionTraces}
        isLoading={!sessionTraceInfosHaveSettled || isLoadingSessionTraceData}
        error={sessionTraceInfosError ?? (sessionTraces.length === 0 ? sessionTraceDataError : undefined)}
        getAssessmentTitle={(assessmentName) => assessmentName}
        onViewTrace={(traceInfo) => {
          setSessionTraceNavigationSessionId(sessionId);
          onSelectTrace(idOf(traceInfo), traceInfo);
        }}
      />
    ) : undefined;
  const body = sessionBody ?? traceBody;

  // Header title mirrors v3: the trace's input preview (falling back to the id) rather than the raw
  // id. Prefer the row's info (present for on-page rows), else the fetched trace's V3 info (covers
  // deep-links not on the current page); while that fetch is in flight, show a skeleton.
  const fetchedInfo = traceData?.info;
  const titleInfo = currentInfo ?? (fetchedInfo && isV3ModelTraceInfo(fetchedInfo) ? fetchedInfo : undefined);
  const renderModalTitle = () => {
    if (drawerViewMode === 'session' && sessionId) {
      return <Typography.Text bold>{sessionId}</Typography.Text>;
    }
    if (titleInfo) {
      return <EvaluationsReviewDetailsHeader evaluationResult={convertTraceInfoV3ToRunEvalEntry(titleInfo)} />;
    }
    if (isLoading) {
      return <GenericSkeleton css={{ width: 200, height: theme.general.heightBase }} />;
    }
    // No trace info resolved (e.g. load error) — fall back to the raw id so the header isn't empty.
    return <Typography.Text bold>{traceId}</Typography.Text>;
  };

  return (
    <>
      <ModelTraceExplorerDrawer
        handleClose={onClose}
        selectPreviousEval={selectPrevious}
        selectNextEval={selectNext}
        isPreviousAvailable={previousAvailable}
        isNextAvailable={nextAvailable}
        headerBanner={sessionTraceHeaderBanner}
        navigationLabel={sessionTurnNavigationLabel}
        renderModalTitle={renderModalTitle}
        isLoading={drawerViewMode === 'trace' && isLoading}
        experimentId={experimentId}
        traceInfo={currentInfo}
        drawerViewMode={drawerViewMode}
        onDrawerViewModeChange={viewModeControl && !isSessionTurnTraceView ? handleDrawerViewModeChange : undefined}
        sessionId={sessionId}
        sessionMetrics={{
          tokenUsage:
            typeof sessionMetrics.sessionTokens.total_tokens === 'number'
              ? {
                  ...sessionMetrics.sessionTokens,
                  total_tokens: sessionMetrics.sessionTokens.total_tokens,
                }
              : undefined,
          cost: sessionMetrics.sessionCost,
          latencySeconds: sessionMetrics.sessionLatency,
          goal: sessionMetrics.goal,
          persona: sessionMetrics.persona,
        }}
        shareUrl={drawerViewMode === 'session' ? sessionShareUrl : undefined}
        sessionNavigationEnabled={!isSessionView || Boolean(sessionNavigation)}
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

import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Global, keyframes } from '@emotion/react';
import {
  ApplyDesignSystemContextOverrides,
  Drawer,
  type InputRef,
  ColumnSplitIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl, FormattedMessage } from 'react-intl';
import {
  createTraceV4LongIdentifier,
  SESSION_ID_METADATA_KEY,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { doesTraceSupportV4API, MLFLOW_SOURCE_RUN_KEY, RunName } from '@databricks/web-shared/genai-traces-table';
import {
  EMPTY_FILTER_MODEL,
  TRACE_COLUMN_IDS,
  TracesErrorAlert,
  type SessionHrefGetter,
  type SessionSelectionHandler,
  type TraceColumnId,
  type TraceHrefGetter,
  type TracesTableViewProps,
  type TracesTableViewState,
} from '@databricks/web-shared/traces-table';
import { useDeleteTracesMutation } from '@mlflow/mlflow/src/experiment-tracking/components/evaluations/hooks/useDeleteTraces';
import {
  AssistantAwareDrawer,
  shouldPreventDrawerDismiss,
} from '@mlflow/mlflow/src/common/components/AssistantAwareDrawer';
import { useAssistant, useCoverFloatingBottomObstructions } from '@mlflow/mlflow/src/assistant';
import { useExperimentPageLayoutContainer } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-page-tabs/ExperimentPageLayout';
import { AssistantAwareActionBar } from '@mlflow/mlflow/src/common/components/AssistantAwareActionBar';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { useNavigate, useSearchParams, useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { shouldEnableIssueDetection, shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import {
  SELECTED_TRACE_ID_QUERY_PARAM,
  TRACE_DRAWER_SESSION_ID_QUERY_PARAM,
  TRACE_DRAWER_VIEW_MODE_QUERY_PARAM,
} from '@mlflow/mlflow/src/experiment-tracking/constants';
// Reuse the generic (branding-free) "/" hotkey hook from datasets-v2.
import { useSlashFocusSearch } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-evaluation-datasets-v2/hooks/useSlashFocusSearch';
import { isAssessmentColumnId } from '../utils/assessmentColumns';
import { isCustomTraceColumnId } from '../utils/customColumns';
import { useTracesV4Controller } from '../hooks/useTracesV4Controller';
import { useTracesV4Density } from '../hooks/useTracesV4Density';
import { useTracesV4Notifications } from '../hooks/useTracesV4Notifications';
import { useTracesV4TraceActions } from '../hooks/useTracesV4TraceActions';
import { useTracesV4UnifiedSearch } from '../hooks/useTracesV4UnifiedSearch';
import { useTracesV4DrawerSessionUrlState } from '../hooks/useTracesV4DrawerSessionUrlState';
import { TracesV4TableSurface } from './TracesV4TableSurface';
import type { TracesV4TraceDrawerSession } from './TracesV4TraceDrawer';
import { useTracesV4ToolbarSlots } from './TracesV4Toolbar';
import { TracesV4DeleteModal } from './TracesV4DeleteModal';
import { makeTracesV4ErrorDescription } from './TracesV4States';
import { TracesV4EmptyState } from './TracesV4EmptyState';
import { IssueDetectionModal } from '../../traces-v3/IssueDetectionModal';
import { useLatestIssueDetection } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-issues/useIssueDetection';
import { useCancelJob } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useCancelJob';
import { isJobComplete } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useFetchJobStatus';
import { TracesV4SavedViewsButton, useTracesV4SavedViews } from './TracesV4SavedViews';
import { type TraceColumnHeaderAction } from '@databricks/web-shared/traces-table';
import { useExperimentPageRevampContext } from '../../../../../pages/experiment-page-tabs/ExperimentPageRevampContext';

interface TracesV4PageContentProps {
  experimentId: string;
}

// Drawer parameters a session link drops: the destination opens the session's own drawer state.
const SESSION_LINK_DROPPED_PARAMS = [
  'traceId',
  'spanId',
  'selectedEvaluationId',
  SELECTED_TRACE_ID_QUERY_PARAM,
  TRACE_DRAWER_VIEW_MODE_QUERY_PARAM,
  TRACE_DRAWER_SESSION_ID_QUERY_PARAM,
];

const PREVIEW_LINE_CLAMP_BY_DENSITY = {
  small: 1,
  standard: 2,
  tall: 6,
} as const;

// Narrows a column id to a standard `TraceColumnId` (assessment columns are namespaced separately).
const isStandardColumnId = (id: string): id is TraceColumnId => (TRACE_COLUMN_IDS as readonly string[]).includes(id);

const SCOPED_DRAWER_TRANSITION_DURATION_MS = 100;

const TracesV4DrawerScopeContext = createContext<HTMLElement | null>(null);

interface DrawerScopeBounds {
  top: number;
  left: number;
  width: number;
  height: number;
}

const measureDrawerScope = (drawerScope: HTMLElement | null): DrawerScopeBounds => {
  const scopeBounds = drawerScope?.getBoundingClientRect();
  return {
    top: scopeBounds?.top ?? 0,
    left: scopeBounds?.left ?? 0,
    width: scopeBounds?.width ?? window.innerWidth,
    height: scopeBounds?.height ?? window.innerHeight,
  };
};

const areDrawerScopeBoundsEqual = (first: DrawerScopeBounds, second: DrawerScopeBounds): boolean =>
  first.top === second.top &&
  first.left === second.left &&
  first.width === second.width &&
  first.height === second.height;

const resolveScopedDrawerWidth = (width: number | string | undefined, scopeWidth: number): number | string => {
  if (typeof width === 'string' && width.endsWith('%')) {
    return (parseFloat(width) / 100) * scopeWidth;
  }
  return width ?? scopeWidth;
};

const getDocumentBody = () => document.body;

// Radix dispatches focus-outside dismissals under this event type; pointer dismissals use a different one.
const RADIX_FOCUS_OUTSIDE_EVENT = 'dismissableLayer.focusOutside';

interface ResponsiveTraceDrawerContentProps extends Drawer.DrawerContentProps {
  isFullscreen?: boolean;
}

type DrawerScopeTransitionDirection = 'entering' | 'leaving' | null;

const scopedDrawerEnterAnimation = keyframes({
  from: { opacity: 0, transform: 'translate(100%, 0)' },
  to: { opacity: 1, transform: 'translate(0, 0)' },
});

const scopedDrawerExitAnimation = keyframes({
  from: { opacity: 1, transform: 'translate(0, 0)' },
  to: { opacity: 0, transform: 'translate(100%, 0)' },
});

/**
 * New eval runs UI: renders the trace drawer inside a body-level clipping container that tracks the
 * experiment page while the MLflow Assistant is open, so the drawer fills the page beside the
 * Assistant panel instead of covering it. The portal stays mounted while the Assistant opens or
 * closes, so the drawer resizes in place without replaying its enter animation.
 */
const ResponsiveTraceDrawerContent = (props: ResponsiveTraceDrawerContentProps) => {
  const { theme } = useDesignSystemTheme();
  const drawerScope = useContext(TracesV4DrawerScopeContext);
  const { isFullscreen = false, css: drawerCss, width, onInteractOutside, title, ...drawerProps } = props;
  const [scopeBounds, setScopeBounds] = useState(() => measureDrawerScope(drawerScope));
  const [drawerPortalContainer, setDrawerPortalContainer] = useState<HTMLDivElement | null>(null);
  const previousDrawerScopeRef = useRef(drawerScope);
  const [suppressEnterAnimation, setSuppressEnterAnimation] = useState(false);
  const [scopeTransitionDirection, setScopeTransitionDirection] = useState<DrawerScopeTransitionDirection>(null);
  const getPopupContainer = useCallback(() => drawerPortalContainer ?? document.body, [drawerPortalContainer]);

  // Radix replaces its internal dialog when modal behavior changes. Disable the replacement's
  // enter animation before paint so opening or closing the Assistant reads as one continuous resize.
  useLayoutEffect(() => {
    if (drawerScope !== previousDrawerScopeRef.current) {
      previousDrawerScopeRef.current = drawerScope;
      setSuppressEnterAnimation(true);
      const nextBounds = measureDrawerScope(drawerScope);
      setScopeTransitionDirection(
        window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ||
          areDrawerScopeBoundsEqual(scopeBounds, nextBounds)
          ? null
          : drawerScope
            ? 'entering'
            : 'leaving',
      );
    }
  }, [drawerScope, scopeBounds]);

  // A stable body-level clipping container lets the bounds transition while preventing the
  // translated drawer from becoming visible over the adjacent Assistant panel.
  useEffect(() => {
    const updateScopeBounds = () => {
      const nextBounds = measureDrawerScope(drawerScope);
      setScopeBounds((currentBounds) =>
        areDrawerScopeBoundsEqual(currentBounds, nextBounds) ? currentBounds : nextBounds,
      );
    };

    updateScopeBounds();
    const resizeObserver =
      drawerScope && typeof ResizeObserver !== 'undefined' ? new ResizeObserver(updateScopeBounds) : undefined;
    if (drawerScope) {
      resizeObserver?.observe(drawerScope);
    }
    window.addEventListener('resize', updateScopeBounds);
    window.addEventListener('scroll', updateScopeBounds, true);
    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener('resize', updateScopeBounds);
      window.removeEventListener('scroll', updateScopeBounds, true);
    };
  }, [drawerScope]);

  const drawerWidth =
    scopeTransitionDirection === 'entering' ? '100%' : resolveScopedDrawerWidth(width, scopeBounds.width);
  const openCloseDuration = theme.animation.transitionDuration;
  const drawerContent = drawerPortalContainer ? (
    <ApplyDesignSystemContextOverrides getPopupContainer={getPopupContainer}>
      {/* The drawer is non-modal here, so keep its overlay transparent like AssistantAwareDrawer does. */}
      <Global styles={{ '[data-testid="drawer-overlay"]': { backgroundColor: 'transparent !important' } }} />
      <Drawer.Content
        {...drawerProps}
        // The header's tooltips and menus render at body level, as in managed, so this clipping container
        // can't cut them off at the drawer's edge beside the Assistant. The drawer body keeps its popups here.
        title={
          <ApplyDesignSystemContextOverrides getPopupContainer={getDocumentBody}>
            {title}
          </ApplyDesignSystemContextOverrides>
        }
        data-trace-drawer-fullscreen={isFullscreen}
        data-trace-drawer-scope-transition={Boolean(scopeTransitionDirection)}
        // Same rule as AssistantAwareDrawer.Content, which this replaces: clicks in the Assistant, on the resize
        // handle or in a body-level popover are not outside the drawer. The drawer is non-modal, so focus can
        // also leave it without a click, e.g. when a menu that opened it hands focus back to its trigger; only
        // a click outside closes it.
        onInteractOutside={(event) => {
          if (event.type === RADIX_FOCUS_OUTSIDE_EVENT || shouldPreventDrawerDismiss(event.target as HTMLElement)) {
            event.preventDefault();
          }
          onInteractOutside?.(event);
        }}
        width={drawerWidth}
        css={[
          { boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04)' },
          drawerCss,
          {
            top: 0,
            right: 0,
            width: drawerWidth,
            maxWidth: scopeTransitionDirection === 'entering' ? '100%' : scopeBounds.width,
            height: '100%',
            maxHeight: '100%',
            pointerEvents: 'auto',
            '@media (prefers-reduced-motion: no-preference)': {
              '&[data-state="open"]': {
                animation: suppressEnterAnimation
                  ? 'none'
                  : `${scopedDrawerEnterAnimation} ${openCloseDuration}ms ease-out forwards`,
              },
              '&[data-state="closed"]': {
                animation: `${scopedDrawerExitAnimation} ${openCloseDuration}ms ease-in forwards`,
              },
            },
          },
        ]}
      />
    </ApplyDesignSystemContextOverrides>
  ) : null;

  return createPortal(
    <div
      ref={setDrawerPortalContainer}
      data-testid="trace-drawer-portal"
      style={{
        top: scopeBounds.top,
        left: scopeBounds.left,
        width: scopeBounds.width,
        height: scopeBounds.height,
      }}
      onTransitionEnd={(event) => {
        if (event.target === event.currentTarget) {
          setScopeTransitionDirection(null);
        }
      }}
      css={{
        position: 'fixed',
        overflow: 'hidden',
        contain: 'layout paint',
        pointerEvents: 'none',
        zIndex: theme.options.zIndexBase + 2,
        '@media (prefers-reduced-motion: no-preference)': {
          transition: `top ${SCOPED_DRAWER_TRANSITION_DURATION_MS}ms ease-out, left ${SCOPED_DRAWER_TRANSITION_DURATION_MS}ms ease-out, width ${SCOPED_DRAWER_TRANSITION_DURATION_MS}ms ease-out, height ${SCOPED_DRAWER_TRANSITION_DURATION_MS}ms ease-out`,
        },
      }}
    >
      {drawerContent}
    </div>,
    document.body,
  );
};

const ResponsiveTraceDrawer = {
  Root: AssistantAwareDrawer.Root,
  Content: ResponsiveTraceDrawerContent,
};

/**
 * Layout controller for the V4 traces tab. Owns URL/data state via `useTracesV4Controller` and feeds
 * the shared `TracesTableView`, mapping the controller's flags to a single `viewState`. The two
 * providers (ModelTraceExplorer, GenAITracesTable), the drawer, the delete modal, and notifications
 * stay MLflow-side.
 */
export const TracesV4PageContent = ({ experimentId }: TracesV4PageContentProps) => {
  const { theme } = useDesignSystemTheme();
  const { enabled: revampEnabled } = useExperimentPageRevampContext();
  const intl = useIntl();
  const navigate = useNavigate();
  const { pathname, search, hash } = useLocation();
  const { notify, notificationContainer } = useTracesV4Notifications();
  const searchInputRef = useRef<InputRef>(null);
  useSlashFocusSearch(searchInputRef);

  const controller = useTracesV4Controller({ experimentId });

  // A detection in flight, so this toolbar can report and end it rather than quietly starting a
  // second run on the same experiment.
  const detection = useLatestIssueDetection(experimentId);
  const { cancelJobAsync } = useCancelJob();
  const isDetecting = Boolean(detection.run) && !isJobComplete(detection.status);
  const issueDetectionInProgress = isDetecting
    ? {
        // No job yet means the run is still being created, so there is nothing to cancel.
        isStarting: !detection.jobId,
        onCancel: detection.jobId
          ? async () => {
              try {
                await cancelJobAsync({ jobId: detection.jobId as string, runUuid: detection.run?.info.run_id });
              } finally {
                // Refreshed even when the cancel fails. "Already finalized" means the job
                // reached a terminal state that this client never saw -- a failed status read
                // stops polling -- so the refresh is exactly what unblocks the toolbar, and
                // skipping it on error left Traces stuck until a page reload.
                await detection.refetch();
              }
            }
          : undefined,
      }
    : undefined;
  const {
    url,
    page,
    columns,
    assessments,
    columnSizing,
    traceCount,
    customColumns,
    columnOrder,
    bulk,
    searchInput,
    filterModel,
    flags,
  } = controller;
  const { density, setDensity } = useTracesV4Density(experimentId);

  // One "Reset to defaults" in the column selector clears standard, assessment, and custom overrides
  // plus the reordered column order.
  const resetColumns = useCallback(() => {
    columns.resetToDefaults();
    assessments.reset();
    customColumns.reset();
    columnOrder.reset();
  }, [columns, assessments, customColumns, columnOrder]);

  // What a view captures: visible standard + assessment ids in display order (from the reorder store),
  // so a saved view's `cols` carries reordering, not just membership.
  const effectiveVisibleColumns = useMemo(() => {
    const visible = new Set<string>([...columns.visibleColumns, ...assessments.visibleIds]);
    return columnOrder.columnOrder.filter((id) => visible.has(id));
  }, [columnOrder.columnOrder, columns.visibleColumns, assessments.visibleIds]);

  // Saved views (dirty model): the hook reads/writes view tags, restores a view's columns + order into
  // the user's own store on open, and reports whether the live table has diverged (dirty) so the Views
  // menu can offer Overwrite / Reset. No read-only preview — the table always renders the real columns.
  const savedViews = useTracesV4SavedViews({
    experimentId,
    visibleColumns: effectiveVisibleColumns,
    filterModel,
    setColumns: columns.setColumns,
    resetColumns,
    setFilterModel: controller.setFilterModel,
    assessmentNames: assessments.candidateNames,
    assessmentVisibility: assessments.visibilityByName,
    setAssessmentVisibility: assessments.setVisibility,
    customVisibility: customColumns.visibilityById,
    setCustomVisibility: customColumns.setVisibility,
    setColumnOrder: columnOrder.setColumnOrder,
  });

  const handleHideColumn = useCallback(
    (columnId: string) => {
      if (isCustomTraceColumnId(columnId)) {
        customColumns.toggle(columnId);
      } else if (isAssessmentColumnId(columnId)) {
        assessments.toggle(columnId);
      } else if (isStandardColumnId(columnId)) {
        columns.toggleColumn(columnId);
      }
    },
    [customColumns, assessments, columns],
  );

  const columnHeaderActions = useMemo<Readonly<Partial<Record<string, TraceColumnHeaderAction>>>>(() => {
    const actions: Partial<Record<string, TraceColumnHeaderAction>> = {};
    if (customColumns.tags.selectorOptions.length > 0) {
      actions['tags'] = {
        label: intl.formatMessage({
          defaultMessage: 'Expand tag columns',
          description: 'Column header action for expanding tag columns',
        }),
        icon: <ColumnSplitIcon />,
        onClick: () => customColumns.tags.setAllVisible(true),
      };
    }
    if (customColumns.metadata.selectorOptions.length > 0) {
      actions['metadata'] = {
        label: intl.formatMessage({
          defaultMessage: 'Expand metadata columns',
          description: 'Column header action for expanding metadata columns',
        }),
        icon: <ColumnSplitIcon />,
        onClick: () => customColumns.metadata.setAllVisible(true),
      };
    }
    return actions;
  }, [customColumns.tags, customColumns.metadata, intl]);

  const actions = useTracesV4TraceActions(experimentId, page.traces, page.refetch, notify);
  // With the new eval runs UI, the drawer opens fullscreen within the experiment page (falling back
  // to this tab's content) while the MLflow Assistant is open, so it sits beside the Assistant panel.
  const isNewEvalRunsUi = shouldEnableNewEvalRunsUi();
  const { isPanelOpen: isAssistantPanelOpen } = useAssistant();
  const experimentPageLayoutContainer = useExperimentPageLayoutContainer();
  const [tracesContentContainer, setTracesContentContainer] = useState<HTMLDivElement | null>(null);
  const drawerScope = experimentPageLayoutContainer ?? tracesContentContainer;
  const getDrawerPortalContainer = useCallback(() => drawerScope ?? document.body, [drawerScope]);
  const isDrawerScoped = isNewEvalRunsUi && isAssistantPanelOpen;
  // The new eval runs UI leaves the Assistant button in place over the open trace drawer, which covers the
  // table's pagination bar, so the button should not keep rising above that bar.
  useCoverFloatingBottomObstructions(isNewEvalRunsUi && Boolean(url.traceId));

  // The selection stores the full `ModelTraceInfoV3` per trace (keyed by id), so this is the entire
  // cross-page selection — every bulk action (judges, Genie, add-to-dataset, labeling, review queue)
  // gets its expected input regardless of which page a trace was selected on.
  const selectedTraceInfos = useMemo(() => Array.from(bulk.selected.values()), [bulk.selected]);

  // Combine custom (tag/metadata) and assessment column defs into one stable array — the table's
  // `extraColumns` prop expects a stable reference, so building it inline would rebuild the columns
  // every render.
  const extraColumns = useMemo(
    () => [...customColumns.columnDefs, ...assessments.columnDefs],
    [customColumns.columnDefs, assessments.columnDefs],
  );

  const deleteTracesMutation = useDeleteTracesMutation();
  const [deleteOpen, setDeleteOpen] = useState(false);
  // Snapshot ids when the modal opens so a refetch/clear mid-prompt can't zero the count or
  // turn confirm into a no-op (mirrors the datasets-v2 delete flow).
  const [pendingDeleteIds, setPendingDeleteIds] = useState<string[]>([]);

  const openDelete = useCallback(() => {
    // Delete needs ids only; the selection Map iterates entries, so take its keys.
    setPendingDeleteIds(Array.from(bulk.selected.keys()));
    setDeleteOpen(true);
  }, [bulk.selected]);

  const cancelDelete = useCallback(() => {
    setDeleteOpen(false);
    setPendingDeleteIds([]);
  }, []);

  const confirmDelete = useCallback(() => {
    if (pendingDeleteIds.length === 0) {
      return;
    }
    const ids = pendingDeleteIds;
    deleteTracesMutation.mutate(
      { experimentId, traceRequestIds: ids },
      {
        onSuccess: () => {
          bulk.clear();
          setDeleteOpen(false);
          setPendingDeleteIds([]);
          page.refetch();
          notify.success(
            intl.formatMessage(
              {
                defaultMessage: 'Deleted {count, plural, one {# trace} other {# traces}}',
                description: 'Success toast after bulk-deleting V4 traces',
              },
              { count: ids.length },
            ),
          );
        },
        // Leave selection intact + the modal open on error so the user can retry.
        onError: (err) => notify.error(err),
      },
    );
  }, [pendingDeleteIds, deleteTracesMutation, experimentId, bulk, page, notify, intl]);

  // UC-backed V4 traces must be opened with a V4 long identifier (`trace:/<catalog.schema>/<id>`);
  // the bare hex id would fall through to the legacy GetTraceInfo path and be rejected as "Invalid
  // request id". Mirrors v1's `useActiveEvaluation` selection logic.
  const handleTraceSelected = useCallback(
    (trace: ModelTraceInfoV3) =>
      url.setTraceId(doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id),
    [url],
  );
  const closeDrawer = useCallback(() => url.setTraceId(undefined), [url]);

  // Open a trace by adding `traceId` to the *current* location rather than resetting to a bare Traces
  // route, so active filters/sort in the URL survive an open (and a Cmd/Ctrl+click into a new tab).
  const getTraceHref = useCallback<TraceHrefGetter>(
    (trace) => {
      const traceId = doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id;
      const searchParams = new URLSearchParams(search);
      searchParams.set('traceId', traceId);
      return `${pathname}?${searchParams.toString()}${hash}`;
    },
    [pathname, search, hash],
  );

  // With the new evaluation runs UI, sessions open in the trace drawer next to the table, so its
  // filters, sort and grouping stay in place.
  const sessionDrawerEnabled = shouldEnableNewEvalRunsUi();
  const drawerSession = useTracesV4DrawerSessionUrlState();
  // Read only by the session drawer's links, so the link builder doesn't change with the URL otherwise.
  const sessionLinkSearch = sessionDrawerEnabled ? search : '';

  // The session cell's only product coupling: build the single-chat-session route (matching v1),
  // deep-linking the current trace via `?selectedTraceId` so the destination opens on that trace.
  // With the session drawer, the link also carries the table's parameters; the route sends it back
  // here with the session open in the drawer.
  const getSessionHref = useCallback<SessionHrefGetter>(
    ({ trace, sessionId }) => {
      const baseUrl = Routes.getExperimentPageTabSingleChatSessionRoute(experimentId, sessionId);
      if (sessionDrawerEnabled) {
        const searchParams = new URLSearchParams(sessionLinkSearch);
        SESSION_LINK_DROPPED_PARAMS.forEach((param) => searchParams.delete(param));
        if (trace.trace_id) {
          searchParams.set(SELECTED_TRACE_ID_QUERY_PARAM, trace.trace_id);
        }
        const query = searchParams.toString();
        return query ? `${baseUrl}?${query}` : baseUrl;
      }
      return trace.trace_id
        ? `${baseUrl}?${new URLSearchParams({ [SELECTED_TRACE_ID_QUERY_PARAM]: trace.trace_id }).toString()}`
        : baseUrl;
    },
    [experimentId, sessionLinkSearch, sessionDrawerEnabled],
  );

  // Clicking a grouped session header row navigates to its single-chat-session view (same route the
  // session cell links to), or opens the session in the drawer with the new evaluation runs UI.
  const { openSession } = drawerSession;
  const handleSessionSelected = useCallback<SessionSelectionHandler>(
    (session) => {
      if (sessionDrawerEnabled) {
        openSession(
          session.sessionId,
          doesTraceSupportV4API(session.trace) ? createTraceV4LongIdentifier(session.trace) : session.trace.trace_id,
        );
        return;
      }
      const href = getSessionHref(session);
      if (href) {
        navigate(href);
      }
    },
    [getSessionHref, navigate, openSession, sessionDrawerEnabled],
  );

  // Sessions on this page in table order, each with its first trace, for the drawer's session arrows.
  const sessionsOnPage = useMemo(() => {
    if (!sessionDrawerEnabled) {
      return [];
    }
    const firstTraceBySession = new Map<string, string>();
    for (const trace of page.traces) {
      const sessionId = trace.trace_metadata?.[SESSION_ID_METADATA_KEY];
      if (sessionId && !firstTraceBySession.has(sessionId)) {
        firstTraceBySession.set(
          sessionId,
          doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace.trace_id,
        );
      }
    }
    return Array.from(firstTraceBySession, ([sessionId, traceId]) => ({ sessionId, traceId }));
  }, [page.traces, sessionDrawerEnabled]);
  const {
    sessionId: drawerSessionId,
    viewMode: drawerViewMode,
    showSessionTrace,
    showSession,
    selectTrace: selectDrawerTrace,
    closeDrawer: closeSessionDrawer,
  } = drawerSession;
  const drawerSessionProps = useMemo<TracesV4TraceDrawerSession | undefined>(() => {
    if (!sessionDrawerEnabled || !drawerSessionId) {
      return undefined;
    }
    const index = sessionsOnPage.findIndex((session) => session.sessionId === drawerSessionId);
    const previous = index > 0 ? sessionsOnPage[index - 1] : undefined;
    const next = index >= 0 && index < sessionsOnPage.length - 1 ? sessionsOnPage[index + 1] : undefined;
    return {
      sessionId: drawerSessionId,
      viewMode: drawerViewMode,
      onViewTrace: showSessionTrace,
      onBackToSession: showSession,
      onPreviousSession: previous ? () => openSession(previous.sessionId, previous.traceId) : undefined,
      onNextSession: next ? () => openSession(next.sessionId, next.traceId) : undefined,
    };
  }, [
    drawerSessionId,
    drawerViewMode,
    openSession,
    sessionDrawerEnabled,
    sessionsOnPage,
    showSession,
    showSessionTrace,
  ]);
  const renderRunName = useCallback(
    (trace: ModelTraceInfoV3) => {
      const runUuid = trace.trace_metadata?.[MLFLOW_SOURCE_RUN_KEY];
      return runUuid ? <RunName experimentId={experimentId} runUuid={runUuid} /> : undefined;
    },
    [experimentId],
  );
  // Filter button "clear all": resets exactly what the count badge totals — the popover clauses AND
  // the URL tag filters — but leaves the search box (a separate control the badge doesn't count).
  const clearAllFilters = useCallback(() => {
    controller.setFilterModel(EMPTY_FILTER_MODEL);
    url.clearTagFilters();
  }, [controller, url]);
  // Empty-state "clear filters": the broader reset that also clears the search query, since the
  // no-results state can be caused by the search too.
  const clearFilters = useCallback(() => {
    searchInput.clear();
    clearAllFilters();
  }, [searchInput, clearAllFilters]);

  const hasRows = page.traces.length > 0;
  const showErrorAlert = Boolean(page.error) && hasRows; // refetch failed but prior rows remain
  // A fetch in flight (first load OR a reload — keepPreviousData keeps isLoading false on reload)
  // renders the skeleton, so error/empty branches wait until the fetch settles.
  const showFirstLoadError = Boolean(page.error) && !hasRows && !page.isFetching;

  const getErrorDescription = useMemo(() => makeTracesV4ErrorDescription(intl), [intl]);

  // AI issue detection: the shared modal (reused from v3) seeds itself with the current bulk
  // selection, falling back to the most-recent page of traces when nothing is selected. Completion
  // toasts are handled globally by `IssueDetectionJobNotifications` (mounted in MlflowRouter).
  const [isIssueDetectionOpen, setIsIssueDetectionOpen] = useState(false);

  // The overview and run pages deep-link here with `?detectIssues=true` to auto-open the modal.
  // Wait for the first page so the modal seeds from real traces, then strip the param via `replace`
  // so a refresh doesn't reopen it.
  const [searchParams, setSearchParams] = useSearchParams();
  useEffect(() => {
    if (!shouldEnableIssueDetection() || searchParams.get('detectIssues') !== 'true' || page.isLoading) {
      return;
    }
    setIsIssueDetectionOpen(true);
    setSearchParams(
      (params) => {
        params.delete('detectIssues');
        return params;
      },
      { replace: true },
    );
  }, [searchParams, setSearchParams, page.isLoading]);

  const selectedTraceIds = useMemo(() => Array.from(bulk.selected.keys()), [bulk.selected]);
  const availableTraceIds = useMemo(
    () => page.traces.map((trace) => trace.trace_id).filter((id): id is string => Boolean(id)),
    [page.traces],
  );

  // The new evaluation runs UI replaces the search box and the Filters button with one search that
  // shows the applied filters as pills and turns typed queries into filters.
  const unifiedSearchEnabled = shouldEnableNewEvalRunsUi();
  const unifiedSearch = useTracesV4UnifiedSearch({
    filterModel,
    setFilterModel: controller.setFilterModel,
    tagFilters: url.tagFilters,
    setTagFilters: url.setTagFilters,
    searchInput,
  });

  const toolbarSlots = useTracesV4ToolbarSlots({
    filterModel: unifiedSearchEnabled ? unifiedSearch.searchFilterModel : filterModel,
    onFilterChange: unifiedSearchEnabled ? unifiedSearch.setSearchFilters : controller.setFilterModel,
    hideFilterButton: unifiedSearchEnabled,
    ...(unifiedSearchEnabled ? { traces: page.traces } : {}),
    onClearFilters: clearAllFilters,
    activeFilterCount: controller.activeFilterCount,
    visibleColumns: columns.visibleColumns,
    onToggleColumn: columns.toggleColumn,
    onResetColumns: resetColumns,
    columnOrder: columnOrder.columnOrder,
    onReorderColumn: columnOrder.reorderColumn,
    assessmentColumns: assessments,
    customColumns,
    sort: url.sort,
    dir: url.dir,
    onSort: url.setSort,
    density,
    onDensityChange: setDensity,
    selectionCount: bulk.selected.size,
    onBulkDelete: openDelete,
    isRefreshing: page.isFetching && !page.isLoading,
    experimentId,
    actions,
    selectedTraceInfos,
    onDetectIssues: () => setIsIssueDetectionOpen(true),
    inProgressIssueDetection: issueDetectionInProgress,
    savedViewsButton: <TracesV4SavedViewsButton experimentId={experimentId} savedViews={savedViews} />,
    isGroupedBySession: controller.isGroupedBySession,
    onToggleSessionGrouping: url.setIsGroupedBySession,
    showSessionViewSwitch: isNewEvalRunsUi,
  });

  // Map the controller's flags to a single shared `viewState`. Order mirrors the prior
  // `renderTableRegion()` precedence: first-load error, then end-of-results, then the two empty
  // states, else the table. The no-warehouse case is handled via `customEmptyState` below.
  const viewState: TracesTableViewState = showFirstLoadError
    ? 'error'
    : !page.isFetching && flags.isEmptyPageBeyondFirst
      ? 'no-more-results'
      : !page.isFetching && flags.hasNoTracesInRange
        ? 'empty'
        : !page.isFetching && flags.hasNoSearchResults
          ? 'no-results'
          : 'ready';

  const tableViewProps: TracesTableViewProps = {
    viewState,
    // Table
    traces: page.traces,
    visibleColumns: columns.visibleColumns,
    extraColumns,
    columnHeaderActions,
    columnOrder: columnOrder.columnOrder,
    initialColumnSizing: columnSizing.columnSizing,
    onColumnSizingSettled: columnSizing.setColumnSizing,
    isLoading: page.isFetching,
    isFetching: page.isFetching,
    skeletonRowCount: url.pageSize,
    onTraceSelected: handleTraceSelected,
    selectedTraceId: url.traceId,
    selectedForBulk: bulk.selected,
    isAllOnPageSelected: bulk.isAllVisibleChecked,
    isSomeOnPageSelected: bulk.isSomeVisibleChecked,
    onToggleBulkRow: bulk.toggle,
    // Session-level select needs the whole session's traces; when a grouped page spans
    // multiple pages the off-page traces aren't loaded, so disable it (mirrors universe).
    onToggleBulkRows: controller.isGroupedBySession && (page.hasNext || page.hasPrev) ? undefined : bulk.toggleMany,
    onToggleBulkAll: bulk.toggleAll,
    sort: url.sort,
    dir: url.dir,
    onSort: url.setSort,
    previewLineClamp: PREVIEW_LINE_CLAMP_BY_DENSITY[density],
    getTraceHref,
    getSessionHref,
    onSessionSelected: handleSessionSelected,
    onFilterByTag: controller.onFilterByTag,
    renderRunName,
    onHideColumn: handleHideColumn,
    isGroupedBySession: controller.isGroupedBySession,
    // Session-level aggregates (e.g. token totals) may be incomplete when off-page traces
    // exist — suppress them for the same reason onToggleBulkRows is disabled above.
    sessionsMayBeIncomplete: controller.isGroupedBySession && (page.hasNext || page.hasPrev),
    onReorderColumn: columnOrder.reorderColumn,
    // Toolbar slots (built by useTracesV4ToolbarSlots) + banner slot
    searchValue: searchInput.input,
    onSearchChange: searchInput.setInput,
    onSearchClear: searchInput.clear,
    onSearchSubmit: searchInput.submit,
    searchInputRef,
    ...(unifiedSearchEnabled
      ? {
          expandSearch: true,
          searchMinimumWidth: unifiedSearch.searchMinimumWidth,
          searchControl: unifiedSearch.renderSearchControl({
            renderFilterEditor: toolbarSlots.renderFilterEditor,
            traces: page.traces,
            inputRef: searchInputRef,
            resetKey: experimentId,
          }),
        }
      : {}),
    leftControls: toolbarSlots.leftControls,
    rightControls: toolbarSlots.rightControls,
    bannerSlot: (
      <>
        {actions.runJudges?.JudgesStatusBanner}
        {showErrorAlert && (
          <TracesErrorAlert error={page.error} onRetry={page.refetch} getErrorDescription={getErrorDescription} />
        )}
      </>
    ),
    // Pagination
    pageIndex: url.pageIndex,
    pageSize: url.pageSize,
    onPageChange: controller.goToPage,
    onPageSizeChange: url.setPageSize,
    hasNext: page.hasNext,
    hasPrev: page.hasPrev,
    // Grouped mode fetches one big page, so the page-size selector is moot. The bar itself stays
    // even when that page holds everything: it carries the session count.
    hidePageSizeSelector: controller.isGroupedBySession,
    // "{n} of {total}" footer count (bottom-left), led by the session count when grouped by session.
    traceCount: controller.traceCount.currentCount,
    traceTotal: controller.traceCount.totalCount,
    sessionCount: controller.traceCount.sessions?.currentCount,
    sessionTotal: controller.traceCount.sessions?.totalCount,
    isTraceCountLoading: controller.traceCount.isTotalLoading,
    // Reserve the pinned pagination bar's height with the floating-obstruction store so the
    // Assistant FAB rises above it instead of overlapping the prev/next/page-size controls.
    PaginationBarWrapper: AssistantAwareActionBar,
    // States
    onClearFilters: clearFilters,
    onRetry: page.refetch,
    error: page.error,
    getErrorDescription,
    customEmptyState:
      // Short-circuits before the viewState switch (keeping toolbar + banner). Gated on the
      // resolved `empty` state, not `hasNoTracesInRange` alone, so error / no-results /
      // no-more-results still flow through the switch — a trace-id-search miss lands on
      // `no-results`, and a first-load error on `error`, not the quickstart.
      viewState === 'empty' ? (
        <TracesV4EmptyState
          kind={controller.emptyStateKind}
          timeLabel={controller.timeLabel}
          onViewAll={() => controller.setTimeRange({ timeLabel: 'ALL' })}
        />
      ) : undefined,
  };

  const content = (
    // The two providers expose the shared trace-action building blocks (drawer add-to-dataset
    // and the dataset modal) to the drawer and Actions menu. The negative bottom margin bleeds the
    // pinned pagination bar into the shared content wrapper's 8px bottom padding, aligning it with the
    // sidebar.
    <TracesV4TableSurface
      providerProps={{
        modelTraceExplorer: {
          renderExportTracesToDatasetsModal: actions.renderExportTracesToDatasetsModal,
          renderAddToReviewQueueDropdown: actions.AddToReviewQueueDropdown,
          DrawerComponent: isNewEvalRunsUi ? ResponsiveTraceDrawer : AssistantAwareDrawer,
          ...(isNewEvalRunsUi
            ? {
                drawerWidth: isDrawerScoped ? '90%' : undefined,
                getDrawerPortalContainer: isDrawerScoped ? getDrawerPortalContainer : undefined,
                initiallyFullscreen: isDrawerScoped,
              }
            : {}),
        },
        genAITracesTable: {
          experimentId,
          getTrace: actions.getTrace,
          isGroupedBySession: controller.isGroupedBySession,
        },
      }}
      tableViewProps={tableViewProps}
      drawerProps={{
        traceId: url.traceId,
        onClose: sessionDrawerEnabled ? closeSessionDrawer : closeDrawer,
        experimentId,
        traces: page.traces,
        onSelectTrace: sessionDrawerEnabled ? selectDrawerTrace : url.setTraceId,
        runJudgeConfiguration: actions.runJudges?.runJudgeConfiguration,
        ...(drawerSessionProps ? { session: drawerSessionProps } : {}),
      }}
      layoutProps={{
        paddingTop: revampEnabled ? 0 : theme.spacing.md,
        paddingLeft: revampEnabled ? 0 : theme.spacing.md,
        // With the new eval runs UI the scope wrapper below clips its content, so the wrapper bleeds instead.
        marginBottom: isNewEvalRunsUi ? 0 : -theme.spacing.sm,
      }}
      overlaySlots={{
        deleteModal: (
          <TracesV4DeleteModal
            open={deleteOpen}
            count={pendingDeleteIds.length}
            isLoading={deleteTracesMutation.isLoading}
            error={deleteTracesMutation.error instanceof Error ? deleteTracesMutation.error : undefined}
            onConfirm={confirmDelete}
            onCancel={cancelDelete}
          />
        ),
        actionModals: (
          <>
            {actions.runJudges?.RunJudgesModal}
            {actions.editTags.EditTagsModal}

            {isIssueDetectionOpen && (
              <IssueDetectionModal
                key={experimentId}
                onClose={() => setIsIssueDetectionOpen(false)}
                experimentId={experimentId}
                initialSelectedTraceIds={selectedTraceIds}
                availableTraceIds={availableTraceIds}
              />
            )}
          </>
        ),
        notificationContainer,
      }}
    />
  );

  if (!isNewEvalRunsUi) {
    return content;
  }

  return (
    <div
      ref={setTracesContentContainer}
      data-testid="traces-v4-drawer-scope"
      css={{
        position: 'relative',
        marginBottom: -theme.spacing.sm,
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        overflow: 'hidden',
        contain: 'layout paint',
      }}
    >
      <TracesV4DrawerScopeContext.Provider value={isDrawerScoped ? drawerScope : null}>
        {content}
      </TracesV4DrawerScopeContext.Provider>
    </div>
  );
};

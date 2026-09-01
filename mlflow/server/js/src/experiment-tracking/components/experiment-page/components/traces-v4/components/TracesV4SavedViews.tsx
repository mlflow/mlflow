import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDispatch } from 'react-redux';
import {
  Button,
  ChevronDownIcon,
  DangerModal,
  DropdownMenu,
  Input,
  LayerIcon,
  Modal,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import {
  EMPTY_FILTER_MODEL,
  TRACE_COLUMN_IDS,
  TRACES_TOOLBAR_COLLAPSE_QUERY,
  ToolbarCollapsibleLabel,
  type TraceColumnId,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';

import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { copyToClipboard } from '@mlflow/mlflow/src/common/utils/copyToClipboard';
import { getUUID } from '@mlflow/mlflow/src/common/utils/ActionUtils';
import { textCompressDeflate } from '@mlflow/mlflow/src/common/utils/StringUtils';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import type { ThunkDispatch } from '@mlflow/mlflow/src/redux-types';
import { deleteExperimentTagApi, setExperimentTagApi } from '@mlflow/mlflow/src/experiment-tracking/actions';
import { useArrayMemo } from '@databricks/web-shared/model-trace-explorer';
import { useGetExperimentQuery } from '@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery';
import {
  decodeSavedViewEnvelope,
  deserializePersistedState,
  encodeSavedViewEnvelope,
} from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/savedViewEnvelope';
import { SavedViewsMenu, type SavedViewMenuItem } from '../../saved-views/SavedViewsMenu';
import {
  buildV4ViewQuery,
  captureV4ViewState,
  decodeViewColumns,
  getTraceV4SavedViewIdFromTagKey,
  getTraceV4SavedViewShareUrl,
  getTraceV4SavedViewTagKey,
  type CapturedV4ViewState,
  TRACE_V4_SHARE_URL_PARAM_KEY,
} from '../utils/tracesV4SavedViewState';
import { capturedV4StatesMatch } from '../utils/tracesV4DirtyState';
import { isSupportedFilterClause, useMlflowTraceFilterFields } from '../utils/filterModel';
import {
  getTraceV3SavedViewIdFromTagKey,
  getTraceV3SavedViewTagKey,
  translateV3ViewState,
  type V3SavedViewState,
} from '../utils/tracesV3ViewCompat';
import { DEFAULT_TRACES_V4_TIME_LABEL } from '../utils/timeRange';

/**
 * Saved views for the V4 traces tab. Reuses the shared tag-envelope codec and the
 * {@link SavedViewsMenu} dropdown body. Most view state is URL-first (search, sort, page size, tag
 * filters and time range all live in the URL), so applying a view is largely a navigation to the
 * stored query; the one piece that isn't in the URL — column visibility — is restored into the
 * user's own column store on open. Once a view is applied, the live table can diverge from it as the
 * user edits ("dirty"); the Views menu then offers Overwrite (persist the edits into the view) and
 * Reset (discard the edits, re-applying the stored view). Opening from the menu and opening from a
 * shared link behave identically — there is no read-only preview.
 */

// Experiment-tag values are capped server-side (MAX_EXPERIMENT_TAG_VAL_LENGTH, 20000 chars); a write
// above the ceiling hard-throws rather than truncating, so preflight the encoded length.
const MAX_TAG_VALUE_LENGTH = 20000;

// Client-side cap (mirrors the V3 / runs MAX_SAVED_VIEWS): each view is a tag and `get-experiment`
// returns every tag value, so the count is bounded to keep that payload small.
export const MAX_SAVED_VIEWS = 40;

interface TraceV4SavedViewSummary {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  // Which tag prefix the view is stored under. A legacy V3 view (`v3`) opens (translated to V4
  // state) and deletes in place; overwriting one migrates it to a V4 tag (see `overwriteView`).
  origin: 'v4' | 'v3';
}

/** `dirty` = live table diverges from the active view; `clean` = matches or no view active. */
export type TracesV4ViewDirtyStatus = 'clean' | 'dirty';

interface UseTracesV4SavedViewsParams {
  experimentId: string;
  /** The user's live visible columns — captured into a view on save, and diffed against it for dirty. */
  visibleColumns: TraceColumnId[];
  /** The live popover filter model — captured into a view on save, restored on open, diffed for dirty. */
  filterModel: TraceFilterModel;
  /** Writes an explicit column set into the user's persisted store (used by open / reset). */
  setColumns: (columns: TraceColumnId[]) => void;
  /** Clears column overrides (standard + assessment) back to defaults; used by "Default view". */
  resetColumns: () => void;
  /** Sets the popover filter clauses (React state, not URL-backed); used by open / reset / default. */
  setFilterModel: (next: TraceFilterModel) => void;
  /** Candidate assessment names, so restored assessment-filter clauses validate against live fields. */
  assessmentNames?: string[];
  /**
   * Live assessment-column visibility by name (localStorage-backed, not URL) — captured into a view on
   * save and diffed for dirty. Defaults to an empty map so a page with no assessments captures nothing.
   */
  assessmentVisibility?: Record<string, boolean>;
  /** Restores a view's assessment visibility (overwrites the override map); used by open / reset. */
  setAssessmentVisibility?: (visibility: Record<string, boolean> | undefined) => void;
  /**
   * Live custom-column visibility by id (localStorage-backed, not URL) — captured into a view on
   * save and diffed for dirty. Defaults to an empty map so a page with no custom columns captures nothing.
   */
  customVisibility?: Record<string, boolean>;
  /** Restores a view's custom-column visibility (overwrites the override map); used by open / reset. */
  setCustomVisibility?: (visibility: Record<string, boolean> | undefined) => void;
}

/**
 * Reads / saves / overwrites / deletes / opens named V4 traces views, and reports whether the live
 * table has diverged from the active view (dirty). Tags are read from the Apollo experiment query
 * (the traces route's source of truth) and written via the redux tag thunks; after a write we
 * refetch Apollo so the new view shows up in the list.
 */
export const useTracesV4SavedViews = ({
  experimentId,
  visibleColumns,
  filterModel,
  setColumns,
  resetColumns,
  setFilterModel,
  assessmentNames = [],
  assessmentVisibility = {},
  setAssessmentVisibility,
  customVisibility = {},
  setCustomVisibility,
}: UseTracesV4SavedViewsParams) => {
  const dispatch = useDispatch<ThunkDispatch>();
  const intl = useIntl();
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: experiment, refetch } = useGetExperimentQuery({ experimentId });

  // Validate a stored filter model against the live field set: drop clauses whose field/operator no
  // longer exists (a since-removed field, or an assessment name not on this page) so a restored view
  // can't silently produce wrong results. Applied both on restore and when normalizing the dirty
  // baseline, so an unsupported clause reads as "already dropped" rather than stranding the view dirty.
  // Stabilize the names by content so a fresh `[]`/array identity doesn't churn `filterFields` →
  // `supportedFilters` → the effects that depend on it every render.
  const stableAssessmentNames = useArrayMemo(assessmentNames);
  const filterFields = useMlflowTraceFilterFields(stableAssessmentNames);
  const supportedFilters = useCallback(
    (filters: TraceFilterModel | undefined): TraceFilterModel =>
      (filters ?? EMPTY_FILTER_MODEL).filter((clause) => isSupportedFilterClause(filterFields, clause)),
    [filterFields],
  );

  // KNOWN LIMITATION (mirrors V3): read-only enforcement is deferred — the traces Apollo query does
  // not fetch `allowedActions`, so Save/Delete default to unrestricted (the write fails server-side
  // for a read-only user). Saved views are intended to eventually be allowed for read-only users.
  const canModify = true;

  const views: TraceV4SavedViewSummary[] = useMemo(() => {
    const tags = experiment?.tags ?? [];
    // Collect V4 and legacy V3 views into one list, keyed by id so a view that has been migrated
    // (a V4 tag written over a V3 one, both sharing the id) is de-duped — V4 always wins.
    const byId = new Map<string, TraceV4SavedViewSummary>();
    for (const { key, value } of tags) {
      if (key == null || value == null) {
        continue;
      }
      const v4Id = getTraceV4SavedViewIdFromTagKey(key);
      const v3Id = v4Id === null ? getTraceV3SavedViewIdFromTagKey(key) : null;
      const id = v4Id ?? v3Id;
      if (id === null) {
        continue;
      }
      const origin: 'v4' | 'v3' = v4Id !== null ? 'v4' : 'v3';
      // A V3 tag must never shadow a migrated V4 tag of the same id.
      if (origin === 'v3' && byId.get(id)?.origin === 'v4') {
        continue;
      }
      try {
        // Both V4 and V3 use the same envelope codec (name/createdAt/updatedAt + compressed state);
        // only the inner `state` shape differs, which matters at open time, not for the summary.
        const { name, createdAt, updatedAt } = decodeSavedViewEnvelope(value);
        byId.set(id, { id, name, createdAt, updatedAt, origin });
      } catch {
        // Skip a corrupt tag rather than breaking the list.
      }
    }
    // Most-recently-edited first: overwriting a view floats it to the top.
    return [...byId.values()].sort((a, b) => b.updatedAt - a.updatedAt);
  }, [experiment?.tags]);

  const atCap = views.length >= MAX_SAVED_VIEWS;

  const saveView = useCallback(
    async (name: string) => {
      if (atCap) {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage(
            {
              defaultMessage:
                'This experiment has reached the maximum of {max} saved views. Delete a view before saving a new one.',
              description: 'Error toast shown when saving a traces view once the experiment is at the saved-view cap',
            },
            { max: MAX_SAVED_VIEWS },
          ),
          3,
        );
        return null;
      }
      // Reject a duplicate name (case-insensitive, trimmed) before writing. Best-effort: tags have
      // no server-side uniqueness constraint, so concurrent writers can still both win.
      const normalized = name.trim().toLowerCase();
      if (views.some((view) => view.name.trim().toLowerCase() === normalized)) {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage(
            {
              defaultMessage: 'A view named "{name}" already exists. Choose a different name.',
              description: 'Error toast shown when saving a traces view whose name is already taken',
            },
            { name: name.trim() },
          ),
          3,
        );
        return null;
      }
      // Capture the current URL view params + the live columns, filter model and assessment/custom-column
      // visibility (none of the latter three ride in the URL).
      const state = captureV4ViewState(
        searchParams,
        visibleColumns,
        filterModel,
        assessmentVisibility,
        customVisibility,
      );
      const compressedState = await textCompressDeflate(JSON.stringify(state));
      const id = getUUID();
      const envelope = encodeSavedViewEnvelope(name.trim(), compressedState, Date.now());
      if (envelope.length > MAX_TAG_VALUE_LENGTH) {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage({
            defaultMessage: 'This view is too large to save.',
            description: 'Error toast shown when a saved traces view exceeds the experiment-tag size limit',
          }),
          3,
        );
        return null;
      }
      await dispatch(setExperimentTagApi(experimentId, getTraceV4SavedViewTagKey(id), envelope));
      await refetch();
      return { id, state };
    },
    [
      dispatch,
      experimentId,
      refetch,
      searchParams,
      visibleColumns,
      filterModel,
      assessmentVisibility,
      customVisibility,
      views,
      atCap,
      intl,
    ],
  );

  const deleteView = useCallback(
    async (id: string) => {
      // Delete every tag stored under this id, across both prefixes. Usually there's just one, but a
      // half-migrated view (a V4 tag written over a V3 one whose delete never landed) leaves both —
      // the list de-dupes them (V4 wins), so deleting only the V4 tag would let the V3 twin resurrect
      // the view on the next refetch. Delete whichever of the two actually exists.
      const tags = experiment?.tags ?? [];
      const v4Key = getTraceV4SavedViewTagKey(id);
      const v3Key = getTraceV3SavedViewTagKey(id);
      const keysToDelete = [v4Key, v3Key].filter((key) => tags.some((tag) => tag.key === key));
      // Fall back to the V4 key if the cache somehow lists neither (best-effort; the delete no-ops).
      await Promise.all(
        (keysToDelete.length > 0 ? keysToDelete : [v4Key]).map((key) =>
          dispatch(deleteExperimentTagApi(experimentId, key)),
        ),
      );
      await refetch();
    },
    [dispatch, experimentId, refetch, experiment?.tags],
  );

  // Decode a stored view's tag value into V4 captured state, or null if it's missing/corrupt. Reads
  // a native V4 tag when present; otherwise falls back to the legacy V3 tag of the same id and
  // translates its frozen state shape into V4's (so a V3 view opens through the normal apply path).
  const decodeViewState = useCallback(
    async (id: string): Promise<CapturedV4ViewState | null> => {
      const tags = experiment?.tags ?? [];
      const v4Tag = tags.find(({ key }) => key === getTraceV4SavedViewTagKey(id));
      if (v4Tag?.value != null) {
        try {
          return (await deserializePersistedState(decodeSavedViewEnvelope(v4Tag.value))) as CapturedV4ViewState;
        } catch {
          return null;
        }
      }
      const v3Tag = tags.find(({ key }) => key === getTraceV3SavedViewTagKey(id));
      if (v3Tag?.value != null) {
        try {
          const v3State = (await deserializePersistedState(decodeSavedViewEnvelope(v3Tag.value))) as V3SavedViewState;
          const translated = translateV3ViewState(v3State);
          // V3 column ids don't necessarily all resolve as V4 columns. Normalize `cols` to the
          // resolvable V4 subset (what applyView will actually restore) so an opened V3 view reads
          // clean, not spuriously dirty against ids V4 dropped. Absent when nothing resolves.
          const resolvedCols = decodeViewColumns(translated, TRACE_COLUMN_IDS);
          if (resolvedCols) {
            translated.single.cols = resolvedCols.join(',');
          } else {
            delete translated.single.cols;
          }
          return translated;
        } catch {
          return null;
        }
      }
      return null;
    },
    [experiment?.tags],
  );

  // Activate a view: rewrite the URL query to its state (+ the share key), and restore the two
  // non-URL surfaces — its columns into the user's own column store, and its popover filter model
  // into React state (validated so a clause referencing a since-removed field/operator is dropped).
  // The V4 hooks read the params on the next render, so this IS the applied view. A view with no
  // resolvable columns leaves the user's columns untouched rather than hiding everything.
  const applyView = useCallback(
    (state: CapturedV4ViewState, id: string) => {
      setSearchParams(new URLSearchParams(buildV4ViewQuery(state, id)));
      const columns = decodeViewColumns(state, TRACE_COLUMN_IDS);
      if (columns) {
        setColumns(columns);
      }
      setFilterModel(supportedFilters(state.filters));
      // Restore assessment-column visibility (localStorage, not URL). An older view without the field
      // clears overrides rather than leaving the previous view's visibility applied on top.
      setAssessmentVisibility?.(state.assessmentColumns);
      // Restore custom-column visibility (localStorage, not URL). An older view without the field
      // clears overrides rather than leaving the previous view's visibility applied on top.
      setCustomVisibility?.(state.customColumns);
    },
    [setSearchParams, setColumns, setFilterModel, supportedFilters, setAssessmentVisibility, setCustomVisibility],
  );

  // Return to the default state: drop every view param, clear the non-URL surfaces (columns +
  // popover filters). Time-range label is kept (not dropped) so the default has a window, not empty.
  const resetToDefaultView = useCallback(() => {
    setSearchParams(new URLSearchParams({ startTimeLabel: DEFAULT_TRACES_V4_TIME_LABEL }));
    resetColumns();
    setFilterModel(EMPTY_FILTER_MODEL);
  }, [setSearchParams, resetColumns, setFilterModel]);

  // Apply a saved view by decoding its stored state, then activating it.
  const openView = useCallback(
    async (id: string) => {
      const state = await decodeViewState(id);
      if (!state) {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage({
            defaultMessage: 'This saved view could not be opened.',
            description: 'Error toast shown when a saved traces view fails to decode',
          }),
          3,
        );
        return;
      }
      applyView(state, id);
    },
    [decodeViewState, applyView, intl],
  );

  // Build a shareable link from a view's STORED state, so the link carries the view's own
  // columns/sort/filters — not whatever the user is currently looking at.
  const buildShareUrl = useCallback(
    async (id: string): Promise<string | null> => {
      const state = await decodeViewState(id);
      return state ? getTraceV4SavedViewShareUrl(experimentId, state, id) : null;
    },
    [decodeViewState, experimentId],
  );

  const activeShareKey = searchParams.get(TRACE_V4_SHARE_URL_PARAM_KEY);
  // The active view id is the share key ONLY when it resolves to a view we actually have — so a
  // stale/garbage share key never drives overwrite/reset/dirty against a phantom view.
  const activeViewId = activeShareKey && views.some((view) => view.id === activeShareKey) ? activeShareKey : null;

  // Stored state of the active view, used both to diff for dirty and to restore columns on a
  // cold-loaded link. Null until decoded (or when no view is active). Declared before the callbacks
  // that close over its setter.
  const [activeStoredState, setActiveStoredState] = useState<CapturedV4ViewState | null>(null);

  // Overwrite an existing view in place with the current live state, keeping its id, name and
  // creation time and bumping `updatedAt` (which floats it to the top of the list). Phantom-guarded:
  // `setExperimentTag` is create-or-update, so a deleted/unknown id would silently resurrect the tag.
  // For a legacy V3 view this MIGRATES it: the state is written under the V4 prefix (same id) and the
  // old V3 tag is deleted, so the view is native V4 afterwards and future overwrites are plain writes.
  const overwriteView = useCallback(
    async (id: string) => {
      const existing = views.find((view) => view.id === id);
      if (!existing) {
        return;
      }
      const state = captureV4ViewState(
        searchParams,
        visibleColumns,
        filterModel,
        assessmentVisibility,
        customVisibility,
      );
      const compressedState = await textCompressDeflate(JSON.stringify(state));
      const envelope = encodeSavedViewEnvelope(existing.name, compressedState, existing.createdAt, Date.now());
      if (envelope.length > MAX_TAG_VALUE_LENGTH) {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage({
            defaultMessage: 'This view is too large to save.',
            description: 'Error toast shown when overwriting a saved traces view exceeds the experiment-tag size limit',
          }),
          3,
        );
        return;
      }
      await dispatch(setExperimentTagApi(experimentId, getTraceV4SavedViewTagKey(id), envelope));
      // Migrate: drop the legacy V3 tag now that the V4 one holds the (edited) state under the same id.
      if (existing.origin === 'v3') {
        await dispatch(deleteExperimentTagApi(experimentId, getTraceV3SavedViewTagKey(id)));
      }
      await refetch();
      // The stored view now matches live; update the baseline so the dirty dot clears immediately.
      setActiveStoredState(state);
      Utils.displayGlobalInfoNotification(
        intl.formatMessage(
          {
            defaultMessage: 'View "{name}" updated.',
            description: 'Success toast shown after overwriting a saved traces view with the current state',
          },
          { name: existing.name },
        ),
        3,
      );
    },
    [
      views,
      searchParams,
      visibleColumns,
      filterModel,
      assessmentVisibility,
      customVisibility,
      dispatch,
      experimentId,
      refetch,
      intl,
    ],
  );

  // Discard live edits: re-apply the active view's stored state (which also restores its columns).
  const resetActiveView = useCallback(() => {
    if (activeViewId) {
      openView(activeViewId);
    }
  }, [activeViewId, openView]);

  // Tracks which view's columns have been restored, so a direct-link landing hydrates columns once
  // but later user edits aren't clobbered on every render. Also gates the dirty diff so it never
  // flashes "dirty" against half-restored state.
  const hydratedViewIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (!activeViewId) {
      setActiveStoredState(null);
      hydratedViewIdRef.current = null;
      return;
    }
    let cancelled = false;
    void decodeViewState(activeViewId).then((state) => {
      if (cancelled || !state) {
        return;
      }
      setActiveStoredState(state);
      // Cold-load: a link opened directly carries the query in the URL but not the columns or the
      // popover filter model, so restore both once per view id. Menu-open already restored them via
      // applyView; this is a harmless no-op in that case.
      if (hydratedViewIdRef.current !== activeViewId) {
        hydratedViewIdRef.current = activeViewId;
        const columns = decodeViewColumns(state, TRACE_COLUMN_IDS);
        if (columns) {
          setColumns(columns);
        }
        setFilterModel(supportedFilters(state.filters));
        setAssessmentVisibility?.(state.assessmentColumns);
        setCustomVisibility?.(state.customColumns);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [
    activeViewId,
    decodeViewState,
    setColumns,
    setFilterModel,
    supportedFilters,
    setAssessmentVisibility,
    setCustomVisibility,
  ]);

  // Dirty = the live table (URL view params + columns) diverges from the active view's stored state.
  // Treated as clean until the stored state is loaded AND columns are hydrated, so it never flashes
  // dirty against a half-restored cold-load.
  const dirtyStatus: TracesV4ViewDirtyStatus = useMemo(() => {
    if (!activeViewId || !activeStoredState || hydratedViewIdRef.current !== activeViewId) {
      return 'clean';
    }
    const live = captureV4ViewState(searchParams, visibleColumns, filterModel, assessmentVisibility, customVisibility);
    // Normalize the stored baseline's filters the same way openView restores them (drop unsupported
    // clauses); diffing raw stored filters would mark a view with a since-removed field dirty forever.
    const normalizedStored: CapturedV4ViewState = {
      ...activeStoredState,
      filters: supportedFilters(activeStoredState.filters),
    };
    return capturedV4StatesMatch(live, normalizedStored) ? 'clean' : 'dirty';
  }, [
    activeViewId,
    activeStoredState,
    searchParams,
    visibleColumns,
    filterModel,
    assessmentVisibility,
    customVisibility,
    supportedFilters,
  ]);

  // Opening a saved-view link in a tab whose experiment was loaded before the view was saved reads a
  // stale Apollo tag cache (a client-side nav doesn't refetch). The view still applies (its state
  // rides in the URL), but the Views list/label/dirty-baseline wouldn't reflect it. When the active
  // share key isn't in our list, refetch ONCE for that key so they catch up. Guarded per-key so a
  // genuinely-missing view doesn't loop.
  const refetchedShareKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!activeShareKey) {
      refetchedShareKeyRef.current = null;
      return;
    }
    const isKnown = views.some((view) => view.id === activeShareKey);
    if (!isKnown && refetchedShareKeyRef.current !== activeShareKey) {
      refetchedShareKeyRef.current = activeShareKey;
      refetch();
    }
  }, [activeShareKey, views, refetch]);

  return {
    views,
    canModify,
    atCap,
    saveView,
    deleteView,
    openView,
    applyView,
    overwriteView,
    resetActiveView,
    resetToDefaultView,
    buildShareUrl,
    activeShareKey,
    activeViewId,
    dirtyStatus,
  };
};

export type TracesV4SavedViewsApi = ReturnType<typeof useTracesV4SavedViews>;

const SaveTraceV4ViewModal = ({
  experimentId,
  visible,
  saveView,
  atCap,
  onCancel,
  onSaved,
}: {
  experimentId: string;
  visible: boolean;
  saveView: (name: string) => Promise<{ id: string; state: CapturedV4ViewState } | null>;
  atCap: boolean;
  onCancel: () => void;
  /** Called with the new view's id + state after a successful save, so the caller can make it active. */
  onSaved: (id: string, state: CapturedV4ViewState) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [name, setName] = useState('');
  const [saving, setSaving] = useState(false);
  const [savedUrl, setSavedUrl] = useState<string | null>(null);

  const reset = useCallback(() => {
    setName('');
    setSaving(false);
    setSavedUrl(null);
    onCancel();
  }, [onCancel]);

  const handleSave = useCallback(async () => {
    const trimmed = name.trim();
    if (!trimmed || saving || atCap) {
      return;
    }
    setSaving(true);
    try {
      const result = await saveView(trimmed);
      // saveView returns null (and shows its own toast) on a duplicate/oversized view; stay on the
      // name-entry phase so the user can rename and retry.
      if (!result) {
        return;
      }
      // Activate from the captured state directly — the refetched tags aren't in cache yet this render.
      onSaved(result.id, result.state);
      setSavedUrl(getTraceV4SavedViewShareUrl(experimentId, result.state, result.id));
      Utils.displayGlobalInfoNotification(
        intl.formatMessage(
          {
            defaultMessage: 'View "{name}" saved.',
            description: 'Success toast shown after a traces view is saved',
          },
          { name: trimmed },
        ),
        3,
      );
    } catch {
      Utils.displayGlobalErrorNotification(
        intl.formatMessage({
          defaultMessage: 'Failed to save the view.',
          description: 'Error toast shown when saving a traces view fails',
        }),
        3,
      );
    } finally {
      setSaving(false);
    }
  }, [name, saving, atCap, saveView, onSaved, experimentId, intl]);

  return (
    <Modal
      componentId="mlflow.traces-v4.save_view.modal"
      title={
        <FormattedMessage
          defaultMessage="Save view"
          description="Title of the modal that saves the current traces view and produces a shareable link"
        />
      }
      visible={visible}
      onCancel={reset}
    >
      {savedUrl ? (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Saved to this experiment. Anyone with access can open this view from the link or the Views list."
              description="Confirmation shown after saving a traces view"
            />
          </Typography.Text>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Input componentId="mlflow.traces-v4.save_view.link" value={savedUrl} readOnly />
            <CopyButton copyText={savedUrl} />
          </div>
        </div>
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Save the current columns, filters, sort and time range as a named view, then share it by link."
              description="Explanation shown in the save-traces-view modal describing what a saved view captures"
            />
          </Typography.Text>
          {atCap && (
            <Typography.Text color="error" data-testid="save-trace-v4-view-at-cap-message">
              <FormattedMessage
                defaultMessage="This experiment has reached the maximum of {max} saved views. Delete a view before saving a new one."
                description="Message shown in the save-traces-view modal when the experiment has reached the saved-view cap"
                values={{ max: MAX_SAVED_VIEWS }}
              />
            </Typography.Text>
          )}
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Input
              componentId="mlflow.traces-v4.save_view.name_input"
              data-testid="save-trace-v4-view-name-input"
              placeholder={intl.formatMessage({
                defaultMessage: 'View name',
                description: 'Placeholder for the name input when saving a traces view',
              })}
              value={name}
              onChange={(e) => setName(e.target.value)}
              onPressEnter={handleSave}
              autoFocus
            />
            <Button
              componentId="mlflow.traces-v4.save_view.save_button"
              data-testid="save-trace-v4-view-save-button"
              type="primary"
              loading={saving}
              disabled={!name.trim() || atCap}
              onClick={handleSave}
            >
              <FormattedMessage defaultMessage="Save" description="Button that saves the current traces view" />
            </Button>
          </div>
        </div>
      )}
    </Modal>
  );
};

/**
 * "Views" dropdown for the V4 traces toolbar: browse / open / copy-link / delete saved views, plus a
 * "Save current view..." entry point. The dropdown body is the shared {@link SavedViewsMenu}; this
 * component owns the traces data source, the copy-link clipboard + toast, the delete-confirmation
 * dialog, and the save modal.
 */
export const TracesV4SavedViewsButton = ({
  experimentId,
  savedViews,
}: {
  experimentId: string;
  savedViews: TracesV4SavedViewsApi;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const {
    views,
    canModify,
    atCap,
    saveView,
    deleteView,
    openView,
    applyView,
    overwriteView,
    resetActiveView,
    resetToDefaultView,
    buildShareUrl,
    activeViewId,
    dirtyStatus,
  } = savedViews;
  const [showSaveModal, setShowSaveModal] = useState(false);
  // Held above the dropdown so the confirm dialog survives the dropdown closing on outside-click.
  // Typed as the menu's item shape (id + name are all the confirm dialog needs).
  const [pendingDelete, setPendingDelete] = useState<SavedViewMenuItem | null>(null);
  const activeView = activeViewId ? views.find((view) => view.id === activeViewId) : undefined;
  const isDirty = dirtyStatus === 'dirty';
  // Shared style for the unsaved-edits dot, rendered twice (inline + a collapsed-only twin).
  const dirtyDotStyles = {
    display: 'inline-block',
    width: 6,
    height: 6,
    borderRadius: '50%',
    backgroundColor: theme.colors.blue500,
    marginLeft: theme.spacing.xs,
    flexShrink: 0,
  } as const;

  const handleCopyLink = async (view: SavedViewMenuItem) => {
    const url = await buildShareUrl(view.id);
    if (!url) {
      Utils.displayGlobalErrorNotification(
        intl.formatMessage({
          defaultMessage: 'This saved view could not be shared.',
          description: 'Error toast shown when building a saved traces view share link fails',
        }),
        3,
      );
      return;
    }
    const ok = await copyToClipboard(url);
    if (ok) {
      Utils.displayGlobalInfoNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Link to "{name}" copied — anyone with access can open this view.',
            description: 'Confirmation toast shown after copying a saved traces view share link',
          },
          { name: view.name },
        ),
        3,
      );
    } else {
      Utils.displayGlobalErrorNotification(
        intl.formatMessage({
          defaultMessage: 'Copy failed — clipboard unavailable.',
          description: 'Error toast shown when copying a saved traces view share link fails',
        }),
        3,
      );
    }
  };

  return (
    <>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button
            componentId="mlflow.traces-v4.saved_views.trigger"
            icon={<LayerIcon />}
            endIcon={<ChevronDownIcon />}
            data-testid="trace-v4-saved-views-trigger"
            // Names the button when its label collapses to icon-only.
            aria-label={
              activeView?.name ??
              intl.formatMessage({
                defaultMessage: 'Default view',
                description:
                  'Label for the saved views dropdown in the traces toolbar when no saved view is active (the default, unfiltered state)',
              })
            }
          >
            <ToolbarCollapsibleLabel>
              {activeView ? (
                <span css={{ display: 'inline-flex', alignItems: 'center', minWidth: 0 }}>
                  <span
                    css={{
                      maxWidth: 200,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      // Tint the name while the view has unsaved edits, echoing the dirty dot.
                      color: isDirty ? theme.colors.blue600 : undefined,
                    }}
                  >
                    {activeView.name}
                  </span>
                  {/* Inline dot after the name while expanded. A twin (below) covers the collapsed,
                      icon-only state; only one is visible at a time, so only the twin carries the testid. */}
                  {isDirty && <span aria-hidden css={dirtyDotStyles} />}
                </span>
              ) : (
                <FormattedMessage
                  defaultMessage="Default view"
                  description="Label for the saved views dropdown in the traces toolbar when no saved view is active (the default, unfiltered state)"
                />
              )}
            </ToolbarCollapsibleLabel>
            {/* The inline dot lives inside the collapsible label, so it vanishes with the name when the
                toolbar collapses to icon-only. This twin sits OUTSIDE the label and shows ONLY while
                collapsed (inverse of the label's own container query), keeping the dirty signal visible. */}
            {isDirty && (
              <span
                data-testid="trace-v4-saved-views-dirty-dot"
                css={{
                  display: 'none',
                  [TRACES_TOOLBAR_COLLAPSE_QUERY]: { ...dirtyDotStyles, marginLeft: 0 },
                }}
              />
            )}
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content align="start">
          <SavedViewsMenu
            componentId="mlflow.traces-v4.saved_views"
            testIdPrefix="trace-v4-saved-views"
            views={views}
            canModify={canModify}
            activeViewId={activeViewId}
            onOpen={openView}
            onCopyLink={handleCopyLink}
            onRequestDelete={setPendingDelete}
            onSaveCurrent={() => setShowSaveModal(true)}
            onSelectDefault={resetToDefaultView}
            dirtyViewActive={isDirty}
            activeViewName={activeView?.name}
            onOverwriteActive={activeViewId ? () => overwriteView(activeViewId) : undefined}
            onResetActive={resetActiveView}
          />
        </DropdownMenu.Content>
      </DropdownMenu.Root>
      <DangerModal
        componentId="mlflow.traces-v4.saved_views.delete_confirm"
        visible={Boolean(pendingDelete)}
        onCancel={() => setPendingDelete(null)}
        onOk={() => {
          if (pendingDelete) {
            deleteView(pendingDelete.id);
          }
          setPendingDelete(null);
        }}
        title={
          <FormattedMessage
            defaultMessage="Delete saved view"
            description="Title of the delete-traces-view confirmation"
          />
        }
        okText={<FormattedMessage defaultMessage="Delete" description="Confirm button for deleting a traces view" />}
      >
        <FormattedMessage
          defaultMessage={`Delete "{name}"? This can't be undone.`}
          description="Body of the delete-traces-view confirmation"
          values={{ name: pendingDelete?.name }}
        />
      </DangerModal>
      <SaveTraceV4ViewModal
        experimentId={experimentId}
        visible={showSaveModal}
        saveView={saveView}
        atCap={atCap}
        onCancel={() => setShowSaveModal(false)}
        onSaved={(id, state) => applyView(state, id)}
      />
    </>
  );
};

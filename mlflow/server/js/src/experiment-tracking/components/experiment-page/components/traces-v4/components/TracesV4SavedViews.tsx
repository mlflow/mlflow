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
import { useGetExperimentQuery } from '@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery';
import {
  decodeSavedViewEnvelope,
  deserializePersistedState,
  encodeSavedViewEnvelope,
} from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/savedViewEnvelope';
import { SavedViewsMenu, type SavedViewMenuItem } from '../../saved-views/SavedViewsMenu';
import { SharedViewBanner } from '../../saved-views/SharedViewBanner';
import {
  buildV4ViewQuery,
  captureV4ViewState,
  decodePreviewColumns,
  getTraceV4SavedViewIdFromTagKey,
  getTraceV4SavedViewShareUrl,
  getTraceV4SavedViewTagKey,
  type CapturedV4ViewState,
  TRACE_V4_COLS_PARAM_KEY,
  TRACE_V4_SHARE_URL_PARAM_KEY,
} from '../utils/tracesV4SavedViewState';
import { DEFAULT_TRACES_V4_TIME_LABEL } from '../utils/timeRange';

/**
 * Saved views for the V4 traces tab. Reuses the shared tag-envelope codec, the {@link SavedViewsMenu}
 * dropdown body, and the {@link SharedViewBanner}. Because the V4 tab is URL-first (search, sort,
 * page size, tag filters and time range all live in the URL), applying a view is just a navigation
 * to the stored query — there is no live-state bridge or React-state preview overlay like V3 needs.
 * The one piece of view state that isn't in the URL — column visibility — rides in a `cols` param
 * that doubles as the preview overlay; Override adopts it into the user's own column store, Discard
 * drops it.
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
}

interface UseTracesV4SavedViewsParams {
  experimentId: string;
  /** The user's live visible columns — captured into a view and snapshotted for Override's Undo. */
  visibleColumns: TraceColumnId[];
  /** Adopts an explicit column set into the user's persisted store (used by Override). */
  setColumns: (columns: TraceColumnId[]) => void;
  /** Clears column overrides (standard + assessment) back to defaults; used by "Default view". */
  resetColumns: () => void;
  /** Clears the popover filter clauses (React state, not URL-backed); used by "Default view". */
  setFilterModel: (next: TraceFilterModel) => void;
}

/**
 * Reads / saves / deletes / opens named V4 traces views, and drives the URL-param preview overlay.
 * Tags are read from the Apollo experiment query (the traces route's source of truth) and written
 * via the redux tag thunks; after a write we refetch Apollo so the new view shows up in the list.
 */
export const useTracesV4SavedViews = ({
  experimentId,
  visibleColumns,
  setColumns,
  resetColumns,
  setFilterModel,
}: UseTracesV4SavedViewsParams) => {
  const dispatch = useDispatch<ThunkDispatch>();
  const intl = useIntl();
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: experiment, refetch } = useGetExperimentQuery({ experimentId });

  // KNOWN LIMITATION (mirrors V3): read-only enforcement is deferred — the traces Apollo query does
  // not fetch `allowedActions`, so Save/Delete default to unrestricted (the write fails server-side
  // for a read-only user). Saved views are intended to eventually be allowed for read-only users.
  const canModify = true;

  const views: TraceV4SavedViewSummary[] = useMemo(() => {
    const tags = experiment?.tags ?? [];
    return tags
      .reduce<TraceV4SavedViewSummary[]>((acc, { key, value }) => {
        if (key == null || value == null) {
          return acc;
        }
        const id = getTraceV4SavedViewIdFromTagKey(key);
        if (id === null) {
          return acc;
        }
        try {
          const { name, createdAt } = decodeSavedViewEnvelope(value);
          acc.push({ id, name, createdAt });
        } catch {
          // Skip a corrupt tag rather than breaking the list.
        }
        return acc;
      }, [])
      .sort((a, b) => b.createdAt - a.createdAt);
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
      // Capture the current URL view params + the live columns (which aren't in the URL).
      const state = captureV4ViewState(searchParams, visibleColumns);
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
    [dispatch, experimentId, refetch, searchParams, visibleColumns, views, atCap, intl],
  );

  const deleteView = useCallback(
    async (id: string) => {
      await dispatch(deleteExperimentTagApi(experimentId, getTraceV4SavedViewTagKey(id)));
      await refetch();
    },
    [dispatch, experimentId, refetch],
  );

  // Decode a stored view's tag value into its captured state, or null if it's missing/corrupt.
  const decodeViewState = useCallback(
    async (id: string): Promise<CapturedV4ViewState | null> => {
      const tag = (experiment?.tags ?? []).find(({ key }) => key === getTraceV4SavedViewTagKey(id));
      if (!tag || tag.value == null) {
        return null;
      }
      try {
        return (await deserializePersistedState(decodeSavedViewEnvelope(tag.value))) as CapturedV4ViewState;
      } catch {
        return null;
      }
    },
    [experiment?.tags],
  );

  // Activate a view by rewriting the URL query to its state (+ the share key). The V4 hooks read
  // their params on the next render, so this IS the applied view — no overlay to mount.
  const applyView = useCallback(
    (state: CapturedV4ViewState, id: string) => {
      setSearchParams(new URLSearchParams(buildV4ViewQuery(state, id)));
    },
    [setSearchParams],
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
  const sharedViewActive = activeShareKey !== null;

  // The columns the shared link carries (the preview overlay). Undefined when no `cols` param or
  // nothing resolves → the caller falls back to the user's own columns.
  const rawCols = searchParams.get(TRACE_V4_COLS_PARAM_KEY);
  const previewColumns = useMemo(
    () => (sharedViewActive ? decodePreviewColumns(rawCols, TRACE_COLUMN_IDS) : undefined),
    [sharedViewActive, rawCols],
  );

  // Drop the preview params (share key + cols) while KEEPING the rest of the applied view (search,
  // sort, tag filters, time range) — the user stays where they navigated, just no longer "previewing".
  const clearPreviewParams = useCallback(() => {
    setSearchParams((params) => {
      params.delete(TRACE_V4_SHARE_URL_PARAM_KEY);
      params.delete(TRACE_V4_COLS_PARAM_KEY);
      return params;
    });
  }, [setSearchParams]);

  // Edit the previewed columns WITHOUT persisting: rewrite the `cols` URL param (the preview state),
  // leaving localStorage untouched until Override. Passing null/empty drops the column override
  // (the table then shows the user's own columns) while staying in preview. This backs the "changes
  // aren't saved unless you override" promise for column toggles during a shared-view preview.
  const setPreviewColumns = useCallback(
    (cols: TraceColumnId[] | null) => {
      setSearchParams((params) => {
        if (cols && cols.length > 0) {
          params.set(TRACE_V4_COLS_PARAM_KEY, cols.join(','));
        } else {
          params.delete(TRACE_V4_COLS_PARAM_KEY);
        }
        return params;
      });
    },
    [setSearchParams],
  );

  // Adopt the previewed columns into the user's own persisted store, then exit preview. This is the
  // only write to the user's column store in this flow. When the link carried no resolvable columns
  // (a filter-/sort-only view), there's nothing to adopt — Override just exits preview.
  const override = useCallback(() => {
    if (previewColumns) {
      setColumns(previewColumns);
    }
    clearPreviewParams();
    Utils.displayGlobalInfoNotification(
      intl.formatMessage({
        defaultMessage: 'This view is now your default.',
        description: 'Traces page > shared view > confirmation toast after adopting a shared view as the own view',
      }),
      5,
    );
  }, [previewColumns, setColumns, clearPreviewParams, intl]);

  const discard = useCallback(() => {
    clearPreviewParams();
  }, [clearPreviewParams]);

  // Opening a saved-view link in a tab whose experiment was loaded before the view was saved reads a
  // stale Apollo tag cache (a client-side nav doesn't refetch). The view still applies (its state
  // rides in the URL), but the Views list/label wouldn't reflect it. When the active share key isn't
  // in our list, refetch ONCE for that key so the list/label catch up. Guarded per-key so a
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
    resetToDefaultView,
    buildShareUrl,
    activeShareKey,
    sharedViewActive,
    previewColumns,
    setPreviewColumns,
    override,
    discard,
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
  const {
    views,
    canModify,
    atCap,
    saveView,
    deleteView,
    openView,
    applyView,
    resetToDefaultView,
    buildShareUrl,
    activeShareKey,
  } = savedViews;
  const { sharedViewActive, override, discard } = savedViews;
  const [showSaveModal, setShowSaveModal] = useState(false);
  // Held above the dropdown so the confirm dialog survives the dropdown closing on outside-click.
  const [pendingDelete, setPendingDelete] = useState<TraceV4SavedViewSummary | null>(null);
  const activeView = activeShareKey ? views.find((view) => view.id === activeShareKey) : undefined;

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
                <span css={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {activeView.name}
                </span>
              ) : (
                <FormattedMessage
                  defaultMessage="Default view"
                  description="Label for the saved views dropdown in the traces toolbar when no saved view is active (the default, unfiltered state)"
                />
              )}
            </ToolbarCollapsibleLabel>
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content align="end">
          <SavedViewsMenu
            componentId="mlflow.traces-v4.saved_views"
            testIdPrefix="trace-v4-saved-views"
            views={views}
            canModify={canModify}
            activeViewId={activeShareKey}
            onOpen={openView}
            onCopyLink={handleCopyLink}
            onRequestDelete={setPendingDelete}
            onSaveCurrent={() => setShowSaveModal(true)}
            onSelectDefault={resetToDefaultView}
            sharedViewActive={sharedViewActive}
            onOverrideActive={override}
            onDiscardActive={discard}
            overrideLabel={
              <FormattedMessage
                defaultMessage="Override my view"
                description="Traces Views menu > entry that adopts the applied shared view into the user's own view"
              />
            }
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

/**
 * Banner shown in the V4 traces `bannerSlot` while a shared view is applied. Reuses the shared
 * {@link SharedViewBanner}; Override adopts the shared view's columns into the user's own store,
 * Discard reverts to the user's own view. Renders nothing when no shared view is active.
 */
export const TracesV4SharedViewBanner = ({ savedViews }: { savedViews: TracesV4SavedViewsApi }) => {
  const { sharedViewActive, override, discard } = savedViews;
  if (!sharedViewActive) {
    return null;
  }
  return (
    <div data-testid="trace-v4-shared-view-banner">
      <SharedViewBanner
        componentId="mlflow.traces-v4.shared_view"
        message={
          <FormattedMessage
            defaultMessage="You're viewing a shared view. Changes you make won't be saved unless you override your view."
            description="Traces page > shared view banner > message shown while a shared view is applied"
          />
        }
        onOverride={override}
        overrideLabel={
          <FormattedMessage
            defaultMessage="Override my view"
            description="Traces page > shared view banner > button that adopts the shared view into the user's own view"
          />
        }
        onDiscard={discard}
      />
    </div>
  );
};

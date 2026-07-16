import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDispatch } from 'react-redux';
import type { EvaluationsOverviewTableSort } from '@databricks/web-shared/genai-traces-table';
import {
  BookmarkIcon,
  Button,
  DangerModal,
  DropdownMenu,
  Input,
  LinkIcon,
  Modal,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { copyToClipboard } from '@mlflow/mlflow/src/common/utils/copyToClipboard';
import { getUUID } from '@mlflow/mlflow/src/common/utils/ActionUtils';
import { textCompressDeflate } from '@mlflow/mlflow/src/common/utils/StringUtils';
import { useNavigate, useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import type { ThunkDispatch } from '@mlflow/mlflow/src/redux-types';
import { deleteExperimentTagApi, setExperimentTagApi } from '@mlflow/mlflow/src/experiment-tracking/actions';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { useGetExperimentQuery } from '@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery';
import {
  decodeSavedViewEnvelope,
  encodeSavedViewEnvelope,
  inflateSavedViewState,
} from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/savedViewEnvelope';
import { SavedViewsMenu, type SavedViewMenuItem } from '../saved-views/SavedViewsMenu';

/**
 * Saved views for the GenAI Traces tab. Reuses the shared tag-envelope codec and the
 * {@link SavedViewsMenu} / {@link SharedViewBanner} presentational components; this file owns the
 * traces-specific parts: capturing/restoring the URL query state (filters/columns/sort/time-range),
 * reading tags from the Apollo experiment query (rather than the redux slice the runs table uses),
 * the copy-link clipboard + toast, the delete-confirmation dialog, and the save modal.
 */

// Distinct from the runs prefix (`mlflow.sharedViewState.`) so a Traces "Views" list never shows
// runs views and vice-versa.
const TRACE_SAVED_VIEW_TAG_PREFIX = 'mlflow.traceViewState.';
export const TRACE_SHARE_URL_PARAM_KEY = 'traceViewShareKey';

// Experiment-tag values are capped at MAX_EXPERIMENT_TAG_VAL_LENGTH (5000 chars) server-side; a
// write above the ceiling HARD-THROWS in the tracking store rather than truncating, so we preflight
// the encoded envelope length before dispatching.
const MAX_TAG_VALUE_LENGTH = 5000;

// The URL params that make up a Traces view. `filter` is multi-valued (one param per active
// filter); searchQuery / isGroupedBySession are in-memory in TracesV3Logs and not yet captured.
const SINGLE_VALUE_KEYS = ['selectedColumns', 'sort', 'viewState', 'startTimeLabel', 'startTime', 'endTime'] as const;
const MULTI_VALUE_KEYS = ['filter'] as const;

type CapturedTraceViewState = {
  single: Partial<Record<(typeof SINGLE_VALUE_KEYS)[number], string>>;
  multi: Partial<Record<(typeof MULTI_VALUE_KEYS)[number], string[]>>;
};

// With shouldEnableTracesTableStatePersistence() off (the OSS default) the table persists
// columns/sort to local storage, not the URL, so capturing from the URL alone yields an empty view.
// TracesV3Logs pushes the live selectedColumns/tableSort through this context so the save path
// serializes what the user actually sees. Wire format matches the URL params the preview decoder
// reads: columns = comma-joined ids, sort = `key::type::asc`. Falls back to the URL when absent.
const COLUMNS_SEPARATOR = ',';
const SORT_SEPARATOR = '::';

interface TraceLiveViewState {
  selectedColumnIds: string[];
  tableSort: EvaluationsOverviewTableSort | undefined;
}

const TraceLiveViewStateContext = createContext<TraceLiveViewState | null>(null);

export const TraceLiveViewStateProvider = TraceLiveViewStateContext.Provider;

const encodeLiveViewState = (live: TraceLiveViewState): CapturedTraceViewState['single'] => {
  const single: CapturedTraceViewState['single'] = {};
  if (live.selectedColumnIds.length > 0) {
    single.selectedColumns = live.selectedColumnIds.join(COLUMNS_SEPARATOR);
  }
  if (live.tableSort) {
    single.sort = [live.tableSort.key, live.tableSort.type, live.tableSort.asc].join(SORT_SEPARATOR);
  }
  return single;
};

const getTraceSavedViewTagKey = (id: string) => `${TRACE_SAVED_VIEW_TAG_PREFIX}${id}`;
const getTraceSavedViewIdFromTagKey = (key: string) =>
  key.startsWith(TRACE_SAVED_VIEW_TAG_PREFIX) ? key.slice(TRACE_SAVED_VIEW_TAG_PREFIX.length) : null;

const captureTraceViewState = (params: URLSearchParams, live?: TraceLiveViewState | null): CapturedTraceViewState => {
  const single: CapturedTraceViewState['single'] = {};
  SINGLE_VALUE_KEYS.forEach((key) => {
    const value = params.get(key);
    if (value !== null) {
      single[key] = value;
    }
  });
  const multi: CapturedTraceViewState['multi'] = {};
  MULTI_VALUE_KEYS.forEach((key) => {
    const values = params.getAll(key);
    if (values.length > 0) {
      multi[key] = values;
    }
  });
  // Prefer the live table state (columns/sort) over whatever the URL happens to hold: with URL
  // persistence off the URL has no selectedColumns/sort at all, and even when it does the live state
  // is authoritative for what the user currently sees.
  if (live) {
    Object.assign(single, encodeLiveViewState(live));
  }
  return { single, multi };
};

const buildTraceViewQuery = (state: CapturedTraceViewState, viewId: string): string => {
  const params = new URLSearchParams();
  Object.entries(state.single ?? {}).forEach(([key, value]) => {
    if (typeof value === 'string') {
      params.set(key, value);
    }
  });
  Object.entries(state.multi ?? {}).forEach(([key, values]) => {
    (values ?? []).forEach((value) => params.append(key, value));
  });
  params.set(TRACE_SHARE_URL_PARAM_KEY, viewId);
  return params.toString();
};

export const getTraceSavedViewShareUrl = (experimentId: string, state: CapturedTraceViewState, viewId: string) => {
  const route = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces);
  return `${window.location.origin}${window.location.pathname}#${route}?${buildTraceViewQuery(state, viewId)}`;
};

interface TraceSavedViewSummary {
  id: string;
  name: string;
  createdAt: number;
}

/**
 * Reads / saves / deletes / opens named Traces views. Tags are read from the Apollo experiment query
 * (the traces route's source of truth) and written via the redux tag thunks the runs feature already
 * ships; after a write we refetch Apollo so the new view shows up in the list.
 */
const useTraceSavedViews = ({ experimentId }: { experimentId: string }) => {
  const dispatch = useDispatch<ThunkDispatch>();
  const navigate = useNavigate();
  const intl = useIntl();
  const [searchParams] = useSearchParams();
  const liveViewState = useContext(TraceLiveViewStateContext);
  const { data: experiment, refetch } = useGetExperimentQuery({ experimentId });

  // The traces Apollo experiment query does not fetch `allowedActions`, so default to unrestricted —
  // mirroring `canModifyExperiment`'s behavior when `allowedActions` is undefined.
  const canModify = true;

  const views: TraceSavedViewSummary[] = useMemo(() => {
    const tags = experiment?.tags ?? [];
    return tags
      .reduce<TraceSavedViewSummary[]>((acc, { key, value }) => {
        if (key == null || value == null) {
          return acc;
        }
        const id = getTraceSavedViewIdFromTagKey(key);
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

  const saveView = useCallback(
    async (name: string) => {
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
      const state = captureTraceViewState(searchParams, liveViewState);
      const compressedState = await textCompressDeflate(JSON.stringify(state));
      const id = getUUID();
      const envelope = encodeSavedViewEnvelope(name, compressedState, Date.now());
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
      await dispatch(setExperimentTagApi(experimentId, getTraceSavedViewTagKey(id), envelope));
      await refetch();
      return { id, state };
    },
    [dispatch, experimentId, refetch, searchParams, liveViewState, views, intl],
  );

  const deleteView = useCallback(
    async (id: string) => {
      await dispatch(deleteExperimentTagApi(experimentId, getTraceSavedViewTagKey(id)));
      await refetch();
    },
    [dispatch, experimentId, refetch],
  );

  const openView = useCallback(
    async (id: string) => {
      const tag = (experiment?.tags ?? []).find(({ key }) => key === getTraceSavedViewTagKey(id));
      if (!tag || tag.value == null) {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage({
            defaultMessage: 'This saved view no longer exists.',
            description: 'Error toast shown when opening a saved traces view that has been deleted',
          }),
          3,
        );
        return;
      }
      let state: CapturedTraceViewState;
      try {
        // Decoding a stored envelope can throw on a corrupt/incompatible tag value; keep the user on
        // the current view rather than navigating into a broken state.
        state = (await inflateSavedViewState(decodeSavedViewEnvelope(tag.value))) as CapturedTraceViewState;
      } catch {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage({
            defaultMessage: 'This saved view could not be opened.',
            description: 'Error toast shown when a saved traces view fails to decode',
          }),
          3,
        );
        return;
      }
      const route = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces);
      navigate(`${route}?${buildTraceViewQuery(state, id)}`);
    },
    [experiment?.tags, experimentId, navigate, intl],
  );

  // Build a shareable link for an existing saved view by inflating its STORED state, so the copied
  // link carries the view's own columns/sort/filters — not whatever the user is currently looking at.
  const buildShareUrl = useCallback(
    async (id: string): Promise<string | null> => {
      const tag = (experiment?.tags ?? []).find(({ key }) => key === getTraceSavedViewTagKey(id));
      if (!tag || tag.value == null) {
        return null;
      }
      try {
        const state = (await inflateSavedViewState(decodeSavedViewEnvelope(tag.value))) as CapturedTraceViewState;
        return getTraceSavedViewShareUrl(experimentId, state, id);
      } catch {
        return null;
      }
    },
    [experiment?.tags, experimentId],
  );

  const activeShareKey = searchParams.get(TRACE_SHARE_URL_PARAM_KEY);

  return { views, canModify, saveView, deleteView, openView, buildShareUrl, activeShareKey };
};

const SaveTraceViewModal = ({
  experimentId,
  visible,
  onCancel,
  onSaved,
}: {
  experimentId: string;
  visible: boolean;
  onCancel: () => void;
  onSaved: (state: CapturedTraceViewState, id: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { saveView } = useTraceSavedViews({ experimentId });
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
    if (!trimmed || saving) {
      return;
    }
    setSaving(true);
    try {
      const result = await saveView(trimmed);
      // saveView returns null (and shows its own toast) on a duplicate name or an oversized view;
      // stay on the name-entry phase so the user can rename and retry.
      if (!result) {
        return;
      }
      const { id, state } = result;
      setSavedUrl(getTraceSavedViewShareUrl(experimentId, state, id));
      onSaved(state, id);
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
  }, [name, saving, saveView, experimentId, onSaved, intl]);

  return (
    <Modal
      componentId="mlflow.traces.save_view.modal"
      title={
        <FormattedMessage
          defaultMessage="Save & share view"
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
            <Input componentId="mlflow.traces.save_view.link" value={savedUrl} readOnly />
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
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Input
              componentId="mlflow.traces.save_view.name_input"
              data-testid="save-trace-view-name-input"
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
              componentId="mlflow.traces.save_view.save_button"
              data-testid="save-trace-view-save-button"
              type="primary"
              loading={saving}
              disabled={!name.trim()}
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
 * "Views" dropdown for the Traces toolbar: browse / open / copy-link / delete saved views, plus a
 * "Save current view..." entry point. Mirrors the runs ExperimentViewSavedViewsButton — the dropdown
 * body is the shared {@link SavedViewsMenu}; this component owns the traces data source, the
 * copy-link clipboard + toast, the delete-confirmation dialog, and the save modal.
 */
export const TracesV3SavedViewsButton = ({ experimentId }: { experimentId: string }) => {
  const intl = useIntl();
  const { views, canModify, deleteView, openView, buildShareUrl } = useTraceSavedViews({ experimentId });
  const [showSaveModal, setShowSaveModal] = useState(false);
  // Held above the dropdown so the confirm dialog survives the dropdown closing on outside-click:
  // a DangerModal rendered inside DropdownMenu.Content would be torn down when the menu dismisses.
  const [pendingDelete, setPendingDelete] = useState<TraceSavedViewSummary | null>(null);

  // Copy the link and fire a page-level toast rather than a tooltip (easily clipped inside a
  // dropdown). The link carries the view's OWN stored state, not whatever the user is viewing now.
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
            componentId="mlflow.traces.saved_views.trigger"
            icon={<BookmarkIcon />}
            data-testid="trace-saved-views-trigger"
          >
            <FormattedMessage
              defaultMessage="Views"
              description="Label for the saved views dropdown in the traces toolbar"
            />
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content align="end">
          <SavedViewsMenu
            componentId="mlflow.traces.saved_views"
            testIdPrefix="trace-saved-views"
            views={views}
            canModify={canModify}
            onOpen={openView}
            onCopyLink={handleCopyLink}
            onRequestDelete={setPendingDelete}
            onSaveCurrent={() => setShowSaveModal(true)}
          />
        </DropdownMenu.Content>
      </DropdownMenu.Root>
      <DangerModal
        componentId="mlflow.traces.saved_views.delete_confirm"
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
      <SaveTraceViewModal
        experimentId={experimentId}
        visible={showSaveModal}
        onCancel={() => setShowSaveModal(false)}
        onSaved={() => {
          // Keep the modal open so the share-link phase (savedUrl) is shown; the modal resets its
          // own state on cancel/close. Do not close here.
        }}
      />
    </>
  );
};

/**
 * "Share" button for the Traces toolbar. Opens the "Save & share view" modal, which names the
 * current view, persists it as a named saved view, then hands back a link. Sharing is always
 * something you do to a *named* view — there is no anonymous current-state link — so this is just a
 * more discoverable, top-level entry point to the same flow as the Views dropdown's "Save current
 * view…" item.
 */
export const TracesV3ShareButton = ({ experimentId }: { experimentId: string }) => {
  const [showModal, setShowModal] = useState(false);

  return (
    <>
      <Button
        componentId="mlflow.traces.share_current_view"
        icon={<LinkIcon />}
        data-testid="trace-share-button"
        onClick={() => setShowModal(true)}
      >
        <FormattedMessage
          defaultMessage="Share"
          description="Label for the button that opens the save-and-share-view modal in the traces toolbar"
        />
      </Button>
      <SaveTraceViewModal
        experimentId={experimentId}
        visible={showModal}
        onCancel={() => setShowModal(false)}
        onSaved={() => {
          // Keep the modal open so the share-link phase (savedUrl) is shown; it resets on close.
        }}
      />
    </>
  );
};

import { useCallback, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { Button, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { omit } from 'lodash';
import { useDispatch, useSelector } from 'react-redux';
import type { ThunkDispatch } from '../../../../../redux-types';
import Routes from '../../../../routes';
import { CopyButton } from '../../../../../shared/building_blocks/CopyButton';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { createExperimentPageUIState, NON_SHAREABLE_UI_STATE_FIELDS } from '../../models/ExperimentPageUIState';
import { textCompressDeflate } from '../../../../../common/utils/StringUtils';
import Utils from '../../../../../common/utils/Utils';
import { EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY, ExperimentPageTabName } from '../../../../constants';
import { getUUID } from '../../../../../common/utils/ActionUtils';
import { setExperimentTagApi } from '../../../../actions';
import {
  encodeSavedViewEnvelope,
  getSavedViewTagKey,
  listSavedViews,
  toKeyValueEntities,
  type SavedViewSummary,
} from '../../utils/savedViewEnvelope';
import { shouldUseCompressedExperimentViewSharedState } from '../../../../../common/utils/FeatureUtils';
import {
  EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY,
  useExperimentPageViewMode,
} from '../../hooks/useExperimentPageViewMode';
import type { ExperimentViewRunsCompareMode } from '../../../../types';
import { loadExperimentViewState } from '../../utils/persistSearchFacets';

type GetShareLinkModalProps = {
  onCancel: () => void;
  visible: boolean;
  experimentId: string;
  searchFacetsState?: ExperimentPageSearchFacetsState;
  uiState?: ExperimentPageUIState;
};

type ShareableViewState = ExperimentPageSearchFacetsState & ExperimentPageUIState;

// Experiment-tag values are capped at MAX_EXPERIMENT_TAG_VAL_LENGTH (5000 chars) server-side
// (mlflow/utils/validation.py); a write above the ceiling HARD-THROWS in the tracking store rather
// than truncating, so we preflight the encoded envelope length and surface a clear error instead of
// the generic "failed to save" that a rejected write would produce.
const MAX_TAG_VALUE_LENGTH = 5000;

// Client-side cap: each view is a tag and `get-experiment` returns every tag value, so the count
// is bounded to keep that payload small. Best-effort; tags have no server-side count constraint.
export const MAX_SAVED_VIEWS = 40;

// Case-insensitive, trimmed — matches the Views search. Best-effort: tags have no server-side
// uniqueness constraint, so concurrent writers can still both win.
export const viewNameExists = (views: SavedViewSummary[], name: string): boolean => {
  const normalized = name.trim().toLowerCase();
  return views.some((view) => view.name.trim().toLowerCase() === normalized);
};

// Typescript-based test to ensure that the keys of the two states are disjoint.
// If they are not disjoint, the state serialization will not work as expected.
const _arePersistedStatesDisjoint: [
  keyof ExperimentPageSearchFacetsState & keyof ExperimentPageUIState extends never ? true : false,
] = [true];

const serializePersistedState = async (state: ShareableViewState) => {
  const shareableState = omit(state, NON_SHAREABLE_UI_STATE_FIELDS);
  const serialized = JSON.stringify(shareableState);
  if (shouldUseCompressedExperimentViewSharedState()) {
    return textCompressDeflate(serialized);
  }
  return serialized;
};

/**
 * Build a share link that references a saved view by its id. The recipient's reader
 * (`useSharedExperimentViewState`) resolves the id against the experiment's saved-view tag, so the
 * URL carries only the opaque id — never the serialized state blob.
 */
export const getSavedViewShareUrl = (
  experimentId: string,
  viewId: string,
  viewMode?: ExperimentViewRunsCompareMode,
) => {
  // The shared view state is consumed by the runs view, so the link must land on
  // the runs tab; the bare experiment route redirects to Overview and drops the param.
  const route = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Runs);

  const queryParams = new URLSearchParams();
  queryParams.set(EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY, viewId);
  if (viewMode) {
    queryParams.set(EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY, viewMode);
  }

  const params = queryParams.toString();
  const hashParam = `${route}${params?.startsWith('?') ? '' : '?'}${params}`;
  return `${window.location.origin}${window.location.pathname}#${hashParam}`;
};

/**
 * "Save & share view" modal. Saving names the current view (columns, sort, filters, charts),
 * persists it as an experiment tag, and produces a link that references it by id. This supersedes
 * the earlier self-contained URL-embedded share link: sharing is now something you do to a durable,
 * named, deletable view rather than a one-off state dump in the URL.
 */
export const ExperimentGetShareLinkModal = ({
  onCancel,
  visible,
  experimentId,
  searchFacetsState,
  uiState,
}: GetShareLinkModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const dispatch = useDispatch<ThunkDispatch>();
  const [viewMode] = useExperimentPageViewMode();

  const [name, setName] = useState('');
  const [saving, setSaving] = useState(false);
  // Set once the view is saved: the id-based share link to show in phase 2.
  const [savedViewUrl, setSavedViewUrl] = useState<string | null>(null);

  const persistKey = useMemo(() => JSON.stringify([experimentId]), [experimentId]);

  // Read from the same live-updating slice as useSavedViews so a view saved this session counts
  // toward the checks below without a GET_EXPERIMENT refetch.
  const tagsById = useSelector((state: any) => state.entities?.experimentTagsByExperimentId?.[experimentId]);
  const existingViews = useMemo<SavedViewSummary[]>(() => listSavedViews(toKeyValueEntities(tagsById)), [tagsById]);
  const atCap = existingViews.length >= MAX_SAVED_VIEWS;

  // When the live search facets / UI state are passed in (classic experiment view),
  // serialize those directly. The modern tabbed page doesn't plumb them into the
  // header, so read the snapshot persisted in local storage instead.
  const liveState = useMemo<ShareableViewState | null>(
    () => (searchFacetsState && uiState ? { ...searchFacetsState, ...uiState } : null),
    [searchFacetsState, uiState],
  );

  const resetAndCancel = useCallback(() => {
    setName('');
    setSaving(false);
    setSavedViewUrl(null);
    onCancel();
  }, [onCancel]);

  const handleSave = useCallback(async () => {
    const trimmed = name.trim();
    if (!trimmed || saving || atCap) {
      return;
    }
    // Reject a duplicate name before doing any work; stay on name-entry so the user can rename.
    if (viewNameExists(existingViews, trimmed)) {
      Utils.displayGlobalErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'A view named "{name}" already exists. Choose a different name.',
            description: 'Error shown when saving an experiment view whose name is already taken',
          },
          { name: trimmed },
        ),
        3,
      );
      return;
    }
    // Read the current view at save time so changes made after the modal mounted
    // (e.g. resizing a column, then saving) are reflected.
    const state =
      liveState ??
      ({
        ...createExperimentPageSearchFacetsState(),
        ...createExperimentPageUIState(),
        ...loadExperimentViewState(persistKey),
      } as ShareableViewState);

    setSaving(true);
    try {
      const compressedState = await serializePersistedState(state);
      const id = getUUID();
      const envelope = encodeSavedViewEnvelope(trimmed, compressedState, Date.now());
      // Preflight the tag-value size: the backend rejects values over the 5000-char ceiling with a
      // hard error, so catch it here and tell the user why rather than showing a generic failure.
      if (envelope.length > MAX_TAG_VALUE_LENGTH) {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage({
            defaultMessage: 'This view is too large to save. Try hiding some columns or charts, then save again.',
            description: 'Error shown when a saved experiment view exceeds the experiment-tag size limit',
          }),
          3,
        );
        return;
      }
      await dispatch(setExperimentTagApi(experimentId, getSavedViewTagKey(id), envelope));
      setSavedViewUrl(getSavedViewShareUrl(experimentId, id, viewMode));
    } catch (e) {
      // Keep the name-entry phase visible so the user can retry a failed write.
      Utils.logErrorAndNotifyUser('Failed to save the view');
    } finally {
      setSaving(false);
    }
  }, [name, saving, atCap, existingViews, liveState, persistKey, experimentId, dispatch, viewMode, intl]);

  return (
    <Modal
      componentId="mlflow.experiment_page.save_and_share_view.modal"
      title={
        <FormattedMessage
          defaultMessage="Save & share view"
          description="Title of the modal that saves the current experiment view and produces a shareable link"
        />
      }
      visible={visible}
      onCancel={resetAndCancel}
    >
      {savedViewUrl ? (
        // Phase 2: the view was saved — offer the link that references it.
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Saved to this experiment. Anyone with access can open this view from the link or the Views list."
              description="Confirmation shown after saving an experiment view, explaining that the link opens the saved view"
            />
          </Typography.Text>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Input
              componentId="mlflow.experiment_page.save_and_share_view.link"
              data-testid="share-link-input"
              value={savedViewUrl}
              readOnly
            />
            <CopyButton copyText={savedViewUrl} data-testid="share-link-copy-button" />
          </div>
        </div>
      ) : (
        // Phase 1: name the view.
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Save the current column layout, filters, sort and charts as a named view, then share it by link."
              description="Explanation shown in the save-and-share-view modal describing what a saved view captures"
            />
          </Typography.Text>
          {atCap && (
            <Typography.Text color="error" data-testid="save-view-at-cap-message">
              <FormattedMessage
                defaultMessage="This experiment has reached the maximum of {max} saved views. Delete a view before saving a new one."
                description="Message shown in the save-view modal when the experiment has reached the saved-view limit"
                values={{ max: MAX_SAVED_VIEWS }}
              />
            </Typography.Text>
          )}
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Input
              componentId="mlflow.experiment_page.save_and_share_view.name_input"
              data-testid="save-view-name-input"
              placeholder={intl.formatMessage({
                defaultMessage: 'View name',
                description: 'Placeholder for the name input when saving an experiment view',
              })}
              value={name}
              onChange={(e) => setName(e.target.value)}
              onPressEnter={handleSave}
              autoFocus
            />
            <Button
              componentId="mlflow.experiment_page.save_and_share_view.save_button"
              data-testid="save-view-save-button"
              type="primary"
              loading={saving}
              disabled={!name.trim() || atCap}
              onClick={handleSave}
            >
              <FormattedMessage
                defaultMessage="Save"
                description="Button that saves the current experiment view as a named view"
              />
            </Button>
          </div>
        </div>
      )}
    </Modal>
  );
};

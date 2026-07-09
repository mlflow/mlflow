import { useCallback, useMemo } from 'react';
import { Button, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { pick } from 'lodash';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';
import Utils from '../../../../common/utils/Utils';
import {
  EXPERIMENT_PAGE_QUERY_PARAM_IS_PREVIEW,
  EXPERIMENT_PAGE_QUERY_PARAM_KEYS,
} from './useExperimentPageSearchFacets';
import type { ExperimentQueryParamsSearchFacets } from './useExperimentPageSearchFacets';
import { loadExperimentViewState, saveExperimentViewState } from '../utils/persistSearchFacets';
import { serializeFieldsToQueryString } from '../utils/persistSearchFacets.serializers';
import {
  EXPERIMENT_PAGE_UI_STATE_FIELDS,
  NON_SHAREABLE_UI_STATE_FIELDS,
  createExperimentPageUIState,
} from '../models/ExperimentPageUIState';
import type { ExperimentPageUIState } from '../models/ExperimentPageUIState';
import { createExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY } from '../../../constants';

/**
 * Actions for the read-only shared-view banner: override the user's own saved view with the
 * currently displayed shared view, or discard the shared view and restore the user's saved one.
 * Both leave shared mode and strip the share key from the URL so a reload doesn't re-enter it.
 */
export const useSharedViewActions = ({
  experimentIds,
  searchFacets,
  uiState,
  setUIState,
  exitSharedView,
}: {
  experimentIds: string[];
  searchFacets: ExperimentQueryParamsSearchFacets | null;
  uiState: ExperimentPageUIState;
  setUIState: React.Dispatch<React.SetStateAction<ExperimentPageUIState>>;
  exitSharedView: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [, setSearchParams] = useSearchParams();
  const persistKey = useMemo(() => JSON.stringify([...experimentIds].sort()), [experimentIds]);

  // Reset the in-memory view to a previously persisted view and rewrite the URL to match, dropping
  // the share key. Used by both Discard (restore the user's own saved view) and Undo (revert a save).
  const restoreView = useCallback(
    (persisted: Record<string, unknown> | undefined) => {
      setUIState({ ...createExperimentPageUIState(), ...pick(persisted, EXPERIMENT_PAGE_UI_STATE_FIELDS) });
      const facets = {
        ...createExperimentPageSearchFacetsState(),
        ...pick(persisted, EXPERIMENT_PAGE_QUERY_PARAM_KEYS),
      };
      const serializedFacets = serializeFieldsToQueryString(facets);
      // Single param write (facets + key removal together) to avoid racing two setSearchParams calls.
      setSearchParams(
        (params) => {
          Object.entries(serializedFacets).forEach(([key, value]) => params.set(key, value as string));
          params.delete(EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY);
          params.delete(EXPERIMENT_PAGE_QUERY_PARAM_IS_PREVIEW);
          return params;
        },
        { replace: true },
      );
    },
    [setUIState, setSearchParams],
  );

  const handleOverrideSavedView = useCallback(() => {
    const previous = loadExperimentViewState(persistKey);
    // Adopt the shared view's shareable state, but keep the user's own non-shareable prefs (pinned/
    // hidden/expanded runs, autoRefresh): a shared link omits those and the reader resets them to
    // defaults on apply, so saving `uiState` as-is would silently wipe the recipient's preferences.
    saveExperimentViewState(
      { ...searchFacets, ...uiState, ...pick(previous, NON_SHAREABLE_UI_STATE_FIELDS) } as never,
      persistKey,
    );
    exitSharedView();
    // Drop the share key (and any preview flag) so a reload doesn't re-enter a non-persisting mode;
    // keep the current facet params. Mirrors the URL cleanup that `restoreView` does on Discard/Undo.
    setSearchParams(
      (params) => {
        params.delete(EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY);
        params.delete(EXPERIMENT_PAGE_QUERY_PARAM_IS_PREVIEW);
        return params;
      },
      { replace: true },
    );
    Utils.displayGlobalInfoNotification(
      <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <FormattedMessage
          defaultMessage="Your saved view was overridden."
          description="Experiment page > shared view > confirmation toast after overriding the user's saved view with a shared view"
        />
        <Button
          componentId="mlflow.experiment_page.shared_view.override_undo"
          size="small"
          onClick={() => {
            saveExperimentViewState(previous as never, persistKey);
            restoreView(previous);
          }}
        >
          <FormattedMessage
            defaultMessage="Undo"
            description="Experiment page > shared view > undo button on the override confirmation toast"
          />
        </Button>
      </span>,
      5,
    );
  }, [persistKey, searchFacets, uiState, exitSharedView, setSearchParams, restoreView, theme.spacing.sm]);

  const handleDiscardSharedView = useCallback(() => {
    restoreView(loadExperimentViewState(persistKey));
    exitSharedView();
  }, [persistKey, restoreView, exitSharedView]);

  return { handleOverrideSavedView, handleDiscardSharedView };
};

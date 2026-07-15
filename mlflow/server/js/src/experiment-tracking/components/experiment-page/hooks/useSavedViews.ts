import { useCallback, useMemo } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import type { ThunkDispatch } from '../../../../redux-types';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import { deleteExperimentTagApi, getExperimentApi } from '../../../actions';
import Routes from '../../../routes';
import { EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY, ExperimentPageTabName } from '../../../constants';
import type { ExperimentEntity } from '../../../types';
import type { SavedViewSummary } from '../utils/savedViewEnvelope';
import { getSavedViewTagKey, listSavedViews, toKeyValueEntities } from '../utils/savedViewEnvelope';
import { canModifyExperiment } from '../utils/experimentPage.common-utils';

/**
 * Reads, deletes and opens named saved views for a single experiment.
 *
 * Saved views live as experiment tags (see {@link savedViewEnvelope}). The list is read from the
 * redux `experimentTagsByExperimentId` slice (not the `experiment.tags` prop) because that slice is
 * the one updated immediately by `setExperimentTagApi`/`deleteExperimentTagApi` — so a just-saved or
 * just-deleted view is reflected in the dropdown without waiting for a `GET_EXPERIMENT` refetch.
 *
 * Saving a view goes through the "Save & share view" modal (which owns the naming UI and the tag
 * write) rather than this hook, so there is a single serialize-and-write path.
 */
export const useSavedViews = ({ experiment }: { experiment: ExperimentEntity }) => {
  const dispatch = useDispatch<ThunkDispatch>();
  const navigate = useNavigate();
  const experimentId = experiment.experimentId;

  // Read the tags slice directly (rather than via getExperimentTags) so a partially-populated
  // store — e.g. one seeded before the experiment's tags have loaded — yields an empty list
  // instead of throwing.
  const tagsById = useSelector((state: any) => state.entities?.experimentTagsByExperimentId?.[experimentId]);

  const views: SavedViewSummary[] = useMemo(
    () => listSavedViews(toKeyValueEntities(tagsById)).sort((a, b) => b.createdAt - a.createdAt),
    [tagsById],
  );

  const canModify = useMemo(() => canModifyExperiment(experiment), [experiment]);

  const deleteView = useCallback(
    (id: string) => dispatch(deleteExperimentTagApi(experimentId, getSavedViewTagKey(id))),
    [dispatch, experimentId],
  );

  const openView = useCallback(
    async (id: string) => {
      // Opening a saved view reuses the shared-link read path: navigate to the runs tab with the
      // view id in the viewStateShareKey param; useSharedExperimentViewState resolves the tag and
      // renders the read-only View Mode banner, same as following any shared link.
      //
      // The reader resolves the id against `experiment.tags`, which flows from the experimentsById
      // slice — and that slice only updates on GET_EXPERIMENT_API, not on the tag write. Refetch the
      // experiment first so a view saved earlier this session (present in experimentTagsByExperimentId
      // but not yet in experiment.tags) is resolvable instead of erroring as "share key does not exist".
      try {
        await dispatch(getExperimentApi(experimentId));
      } catch {
        // Best-effort refresh; the tag is very often already present (e.g. loaded via GET on page
        // load), so navigate regardless and let the reader surface a genuine not-found.
      }
      const route = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Runs);
      navigate(`${route}?${EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY}=${id}`);
    },
    [dispatch, navigate, experimentId],
  );

  return { views, canModify, deleteView, openView };
};

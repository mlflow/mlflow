import { useEffect } from 'react';
import invariant from 'invariant';
import { useLocation, useNavigate, useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';

const LEGACY_PARAM = 'selectedDatasetId';

/**
 * One-way back-compat shim: rewrites legacy `?selectedDatasetId=<id>` URLs (produced
 * by the legacy Datasets tab) to the V2 detail route, preserving any other query
 * params. Keeps shared URLs from the old UI clickable in the new UI.
 *
 * Why: the legacy tab parked the selected dataset in a query param. The V2 UI uses
 * a route param so the dataset detail is a real page. Without this hook, old links
 * silently land on the list view.
 *
 * Delete this file when the legacy tab is removed.
 */
export const useLegacySelectedDatasetRedirect = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  const search = useLocation().search;
  const navigate = useNavigate();

  // Parse once per location change; cheap, and avoids stale closures.
  const params = new URLSearchParams(search);
  const legacySelectedDatasetId = params.get(LEGACY_PARAM);

  useEffect(() => {
    if (!legacySelectedDatasetId) {
      return;
    }
    // Strip the legacy param so the URL doesn't keep triggering the redirect.
    const next = new URLSearchParams(search);
    next.delete(LEGACY_PARAM);
    const nextSearch = next.toString();
    navigate(
      {
        pathname: Routes.getExperimentPageDatasetDetailRoute(experimentId, legacySelectedDatasetId),
        search: nextSearch ? `?${nextSearch}` : '',
      },
      { replace: true },
    );
  }, [experimentId, legacySelectedDatasetId, navigate, search]);

  return { isRedirecting: Boolean(legacySelectedDatasetId) };
};

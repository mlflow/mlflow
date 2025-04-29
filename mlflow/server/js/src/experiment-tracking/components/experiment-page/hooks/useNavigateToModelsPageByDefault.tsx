// The automatic navigation to models tab by default is disabled for now until we properly handle all the use cases (e.g. AutoML)
// TODO(ML-47062): Re-enable this feature when we have a proper solution.
export const useNavigateToModelsPageByDefault = () => {
  return false;
};

// Previous implementation:
/*
const useNavigateToModelsPageByDefault = () => {
  const experimentIds = useExperimentIds();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const usingLoggedModelListPageByDefault = isExperimentLoggedModelsUIEnabled();
  // Navigate to logged models tab (page) only if:
  const shouldNavigateToModelsPage =
    // 1) Logged models feature UI is enabled
    usingLoggedModelListPageByDefault &&
    // 2) The view mode is not set
    !params.get(EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY) &&
    // 3) There is only one experiment (we're not comparing)
    experimentIds.length === 1;
  useEffect(() => {
    if (shouldNavigateToModelsPage) {
      navigate(Routes.getExperimentLoggedModelListPageRoute(experimentIds[0]), { replace: true });
    }
  }, [navigate, experimentIds, shouldNavigateToModelsPage]);
  return shouldNavigateToModelsPage;
};
*/

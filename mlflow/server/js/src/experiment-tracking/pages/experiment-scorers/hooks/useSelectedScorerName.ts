import { useSearchParams } from '../../../../common/utils/RoutingUtils';
import { shouldEnableNewEvalRunsUi } from '../../../../common/utils/FeatureUtils';

export const SELECTED_SCORER_NAME_QUERY_PARAM_KEY = 'selectedScorerName';

/**
 * The scorer a link asked the scorers page to open, such as a scorer listed on an evaluation run. Only read with
 * the new evaluation runs UI, which is the only place that links here with it.
 */
export const useSelectedScorerName = (): string | undefined => {
  const [searchParams] = useSearchParams();
  const selectedScorerName = searchParams.get(SELECTED_SCORER_NAME_QUERY_PARAM_KEY);
  return shouldEnableNewEvalRunsUi() && selectedScorerName ? selectedScorerName : undefined;
};

import { useParams } from '../../../common/utils/RoutingUtils';
import { RunPageTabName } from '../../constants';

/**
 * Returns the run view's active tab.
 * - Supports multi-slash artifact paths (hence '*' catch-all param)
 * - Supports both new (/artifacts/...) and previous (/artifactPath/...) routes
 */
export const useRunViewActiveTab = (): RunPageTabName => {
  const { '*': tabParam } = useParams<{ '*': string }>();
  if (tabParam === 'model-metrics') {
    return RunPageTabName.MODEL_METRIC_CHARTS;
  }
  if (tabParam === 'system-metrics') {
    return RunPageTabName.SYSTEM_METRIC_CHARTS;
  }
  if (tabParam === 'evaluations') {
    return RunPageTabName.EVALUATIONS;
  }
  if (tabParam === 'traces') {
    return RunPageTabName.TRACES;
  }
  if (tabParam?.match(/^(artifactPath|artifacts)/)) {
    return RunPageTabName.ARTIFACTS;
  }

  return RunPageTabName.OVERVIEW;
};

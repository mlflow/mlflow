import { shouldEnableTracingUI } from '../../../../common/utils/FeatureUtils';

export function useShouldShowCombinedRunsTab() {
  return shouldEnableTracingUI();
}

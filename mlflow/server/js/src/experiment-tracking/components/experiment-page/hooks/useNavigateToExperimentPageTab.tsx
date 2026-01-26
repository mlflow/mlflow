import { useEffect, useMemo } from 'react';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { ExperimentKind, ExperimentPageTabName } from '../../../constants';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';
import { useExperimentKind } from '../../../utils/ExperimentKindUtils';
import { coerceToEnum } from '@databricks/web-shared/utils';
import { shouldEnableExperimentOverviewTab } from '../../../../common/utils/FeatureUtils';
import { useIsFileStore } from '../../../hooks/useTrackingStoreInfo';

/**
 * This hook navigates user to the appropriate tab in the experiment page based on the experiment kind.
 */
export const useNavigateToExperimentPageTab = ({
  enabled,
  experimentId,
}: {
  enabled?: boolean;
  experimentId: string;
}) => {
  const navigate = useNavigate();
  const isFileStore = useIsFileStore();

  const { data: experiment, loading: loadingExperiment } = useGetExperimentQuery({
    experimentId,
    options: {
      skip: !enabled,
    },
  });

  const experimentTags = useMemo(() => {
    if (!experiment) return [];
    return experiment && 'tags' in experiment ? experiment?.tags : [];
  }, [experiment]);

  const experimentKindFromContext = useExperimentKind(experimentTags);

  const experimentKind = useMemo(() => {
    if (loadingExperiment || !experiment) {
      return null;
    }

    if (experimentKindFromContext) {
      return coerceToEnum(ExperimentKind, experimentKindFromContext, ExperimentKind.NO_INFERRED_TYPE);
    }
    return null;
  }, [experiment, loadingExperiment, experimentKindFromContext]);

  useEffect(() => {
    if (!enabled || !experimentKind) {
      return;
    }

    // By default, we navigate to the Runs tab
    let targetTab = ExperimentPageTabName.Runs;

    // For GENAI_DEVELOPMENT, we navigate to the Overview tab if enabled and not using FileStore,
    // otherwise Traces tab.
    if (experimentKind === ExperimentKind.GENAI_DEVELOPMENT) {
      targetTab =
        shouldEnableExperimentOverviewTab() && isFileStore === false
          ? ExperimentPageTabName.Overview
          : ExperimentPageTabName.Traces;
    }

    navigate(Routes.getExperimentPageTabRoute(experimentId, targetTab), { replace: true });
  }, [navigate, experimentId, enabled, experimentKind, isFileStore]);

  return {
    isEnabled: enabled,
    isLoading: enabled && (loadingExperiment || isFileStore === undefined),
  };
};

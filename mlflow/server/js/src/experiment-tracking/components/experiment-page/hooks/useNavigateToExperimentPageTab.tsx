import { useEffect, useMemo } from 'react';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { ExperimentKind, ExperimentPageTabName } from '../../../constants';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';
import { getExperimentKindFromTags } from '../../../utils/ExperimentKindUtils';
import { coerceToEnum } from '@databricks/web-shared/utils';

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

  const { data: experiment, loading: loadingExperiment } = useGetExperimentQuery({
    experimentId,
    options: {
      skip: !enabled,
    },
  });

  const experimentKind = useMemo(() => {
    if (loadingExperiment || !experiment) {
      return null;
    }
    const experimentTags = experiment && 'tags' in experiment ? experiment?.tags : [];

    if (experiment) {
      const experimentKindTagValue = getExperimentKindFromTags(experimentTags);
      return coerceToEnum(ExperimentKind, experimentKindTagValue, ExperimentKind.NO_INFERRED_TYPE);
    }
    return null;
  }, [experiment, loadingExperiment]);

  useEffect(() => {
    if (!enabled || !experimentKind) {
      return;
    }

    // By default, we navigate to the Runs tab
    let targetTab = ExperimentPageTabName.Runs;

    // For GENAI_DEVELOPMENT, we navigate to the Traces tab.
    if (experimentKind === ExperimentKind.GENAI_DEVELOPMENT) {
      targetTab = ExperimentPageTabName.Traces;
    }

    navigate(Routes.getExperimentPageTabRoute(experimentId, targetTab), { replace: true });
  }, [navigate, experimentId, enabled, experimentKind]);

  return {
    isEnabled: enabled,
    isLoading: enabled && loadingExperiment,
  };
};

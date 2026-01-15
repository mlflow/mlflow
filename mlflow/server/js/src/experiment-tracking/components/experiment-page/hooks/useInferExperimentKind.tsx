import { useEffect, useMemo, useState } from 'react';
import { useExperimentContainsTraces } from '../../traces/hooks/useExperimentContainsTraces';
import { ExperimentKind, ExperimentPageTabName } from '../../../constants';
import { useExperimentContainsTrainingRuns } from '../../traces/hooks/useExperimentContainsTrainingRuns';
import { isEditableExperimentKind } from '../../../utils/ExperimentKindUtils';
import { matchPath, useLocation } from '../../../../common/utils/RoutingUtils';
import { RoutePaths } from '../../../routes';

export const useInferExperimentKind = ({
  experimentId,
  isLoadingExperiment,
  enabled = true,
  experimentTags,
  updateExperimentKind,
}: {
  experimentId?: string;
  isLoadingExperiment: boolean;
  enabled?: boolean;
  experimentTags?: { key?: string | null; value?: string | null }[] | null;
  updateExperimentKind: (params: { experimentId: string; kind: ExperimentKind }) => void;
}) => {
  const { containsTraces, isLoading: isTracesBeingDetermined } = useExperimentContainsTraces({
    experimentId,
    enabled,
  });

  const [isDismissed, setIsDismissed] = useState(false);

  const { containsRuns, isLoading: isTrainingRunsBeingDetermined } = useExperimentContainsTrainingRuns({
    experimentId,
    enabled,
  });

  const isLoading = enabled && (isLoadingExperiment || isTracesBeingDetermined || isTrainingRunsBeingDetermined);

  const inferredExperimentKind = useMemo(() => {
    if (!enabled || isLoading || isDismissed) {
      return undefined;
    }
    if (containsTraces) {
      return ExperimentKind.GENAI_DEVELOPMENT_INFERRED;
    }
    if (containsRuns) {
      return ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED;
    }
    return ExperimentKind.NO_INFERRED_TYPE;
  }, [
    // prettier-ignore
    enabled,
    isDismissed,
    isLoading,
    containsTraces,
    containsRuns,
  ]);

  // Check if the user landed on the experiment page without a specific tab in the URL.
  // If the URL already contains a tab name (e.g. /experiments/1/traces), we should not
  // redirect them.
  const { pathname } = useLocation();
  const isOnExperimentPageWithoutTab = Boolean(matchPath(RoutePaths.experimentPage, pathname));

  const inferredExperimentPageTab = useMemo(() => {
    if (!isOnExperimentPageWithoutTab) {
      return undefined;
    }
    if (inferredExperimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED) {
      return ExperimentPageTabName.Overview;
    }
    if (inferredExperimentKind === ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED) {
      return ExperimentPageTabName.Runs;
    }
    return undefined;
  }, [inferredExperimentKind, isOnExperimentPageWithoutTab]);

  // automatically update the experiment type if it's not user-editable
  useEffect(() => {
    if (inferredExperimentKind && !isEditableExperimentKind(inferredExperimentKind)) {
      updateExperimentKind({ experimentId: experimentId ?? '', kind: inferredExperimentKind });
    }
  }, [experimentId, inferredExperimentKind, updateExperimentKind]);

  return {
    isLoading,
    inferredExperimentKind,
    inferredExperimentPageTab,
    dismiss: () => setIsDismissed(true),
  };
};

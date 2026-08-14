import { useMemo } from 'react';
import { useQueries } from '../../../../common/utils/reactQueryHooks';
import type { LoggedModelProto } from '../../../types';
import type { RunPageModelVersionSummary } from '../../run-page/hooks/useUnifiedRegisteredModelVersionsSummariesForRun';
import { createMLflowRoutePath } from '../../../../common/utils/RoutingUtils';
import { isUCModelName } from '../../../utils/IsUCModelName';
const MODEL_VERSIONS_TAG_NAME = 'mlflow.modelVersions';

const getUCModelUrl = (name: string, version: string) =>
  `/explore/data/models/${name.replace(/\./g, '/')}/version/${version}`;
const getWMRModelUrl = (name: string, version: string) => createMLflowRoutePath(`/models/${name}/versions/${version}`);

type RegisteredModelEntry = { name: string; version: string };

const getTagValueForModel = (loggedModel: LoggedModelProto): RegisteredModelEntry[] | null => {
  try {
    const tagValue = loggedModel.info?.tags?.find((tag) => tag.key === MODEL_VERSIONS_TAG_NAME)?.value;

    if (tagValue) {
      // Try to parse the tag. If it's malformed, catch and return nothing.
      const parsed = JSON.parse(tagValue) as { name: string; version: string | number }[];
      return (Array.isArray(parsed) ? parsed : []).map((entry) => ({
        name: String(entry?.name ?? ''),
        version: String(entry?.version ?? ''),
      }));
    }
  } catch (e) {
    return null;
  }
  return null;
};

/**
 * Extracts registered model versions from a logged model.
 * Uses both `registrations` (from API) and `mlflow.modelVersions` tag for consistent display
 * across Runs page, Models page, and Run details - fixing inconsistent display when navigating
 * between pages (see https://github.com/mlflow/mlflow/issues/20671).
 */
const getRegisteredModelVersionsForLoggedModel = (loggedModel: LoggedModelProto): RegisteredModelEntry[] => {
  const fromRegistrations =
    loggedModel.info?.registrations?.map((r) => ({
      name: String(r?.name ?? ''),
      version: String(r?.version ?? ''),
    })) ?? [];
  const fromTags = getTagValueForModel(loggedModel) ?? [];

  // Merge both sources, deduplicating by name+version (registrations take precedence)
  const seen = new Set<string>();
  const result: RegisteredModelEntry[] = [];
  for (const entry of [...fromRegistrations, ...fromTags]) {
    if (!entry.name) continue;
    const key = `${entry.name}:${entry.version}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(entry);
  }
  return result;
};

// Hook for ACL checking logic
const useModelVersionsAclCheck = (
  ucModels: RunPageModelVersionSummary[],
  checkAcl: boolean,
): { aclResults: Record<string, boolean>; isLoading: boolean } => {
  const queries = useMemo(() => {
    if (!checkAcl || ucModels.length === 0) {
      return [];
    }
    return [];
  }, [ucModels, checkAcl]);

  const queryResults = useQueries({ queries });

  const { aclResults, isLoading } = useMemo(() => {
    if (!checkAcl || ucModels.length === 0) {
      return { aclResults: {}, isLoading: false };
    }

    const isLoading = queryResults.some((result) => result.isLoading);
    const aclResults: Record<string, boolean> = {};
    return { aclResults, isLoading };
  }, [
    // prettier-ignore
    queryResults,
    checkAcl,
    ucModels.length,
  ]);

  return { aclResults, isLoading };
};

export interface RunPageModelVersionSummaryWithAccess extends RunPageModelVersionSummary {
  hasAccess: boolean;
}

export interface UseExperimentLoggedModelRegisteredVersionsResult {
  modelVersions: RunPageModelVersionSummaryWithAccess[];
  isLoading: boolean;
}

export const useExperimentLoggedModelRegisteredVersions = ({
  loggedModels,
  checkAcl = false,
}: {
  loggedModels: LoggedModelProto[];
  checkAcl?: boolean;
}): UseExperimentLoggedModelRegisteredVersionsResult => {
  // Combined useMemo for parsing tags/registrations and creating model versions
  const { modelVersions, ucModels } = useMemo(() => {
    const modelVersions = loggedModels.flatMap((loggedModel) => {
      const modelVersionsInTag = getRegisteredModelVersionsForLoggedModel(loggedModel);
      return modelVersionsInTag.map((registeredModelEntry) => {
        const isUCModel = isUCModelName(registeredModelEntry.name);
        const getUrlFn = isUCModel ? getUCModelUrl : getWMRModelUrl;
        return {
          displayedName: registeredModelEntry.name,
          version: registeredModelEntry.version,
          link: getUrlFn(registeredModelEntry.name, registeredModelEntry.version),
          source: null,
          status: null,
          sourceLoggedModel: loggedModel,
        };
      });
    });

    const ucModels = modelVersions.filter((model) => model.displayedName && isUCModelName(model.displayedName));

    return { modelVersions, ucModels };
  }, [loggedModels]);

  const { aclResults, isLoading } = useModelVersionsAclCheck(ucModels, checkAcl);

  // Add hasAccess to each model version
  const modelVersionsWithAccess = useMemo<RunPageModelVersionSummaryWithAccess[]>(
    () =>
      modelVersions.map((modelVersion) => {
        const displayedName = modelVersion.displayedName;
        const isUCModel = displayedName && isUCModelName(displayedName);

        let hasAccess = true; // Default for workspace models

        if (checkAcl && isUCModel && displayedName) {
          // For UC models with ACL check enabled, use the ACL result
          hasAccess = aclResults[`${displayedName}:${modelVersion.version}`] ?? false;
        }

        return {
          ...modelVersion,
          hasAccess,
        };
      }),
    [modelVersions, checkAcl, aclResults],
  );

  return {
    modelVersions: modelVersionsWithAccess,
    isLoading,
  };
};

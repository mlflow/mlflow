import { useSelector } from 'react-redux';
import type { ReduxState } from '../../../../redux-types';
import { ModelRegistryRoutes } from '../../../../model-registry/routes';
import { shouldEnableGraphQLModelVersionsForRunDetails } from '../../../../common/utils/FeatureUtils';
import type { UseGetRunQueryResponse } from './useGetRunQuery';
import type { LoggedModelProto } from '../../../types';

/**
 * A unified model version summary that can be used to display model versions on the run page.
 */
export type RunPageModelVersionSummary = {
  displayedName: string | null;
  version: string | null;
  link: string;
  status: string | null;
  source: string | null;
  sourceLoggedModel?: LoggedModelProto;
};

/**
 * We're currently using multiple ways to get model versions on the run page,
 * we also differentiate between UC and workspace registry models.
 *
 * This hook is intended to unify the way we get model versions on the run page to be displayed in overview and register model dropdown.
 */
export const useUnifiedRegisteredModelVersionsSummariesForRun = ({
  queryResult,
  runUuid,
}: {
  runUuid: string;
  queryResult?: UseGetRunQueryResponse;
}): RunPageModelVersionSummary[] => {
  const { registeredModels: registeredModelsFromStore } = useSelector(({ entities }: ReduxState) => ({
    registeredModels: entities.modelVersionsByRunUuid[runUuid],
  }));

  if (shouldEnableGraphQLModelVersionsForRunDetails()) {
    const result: RunPageModelVersionSummary[] = [];
    if (queryResult?.data && 'modelVersions' in queryResult.data) {
      queryResult.data?.modelVersions?.forEach((modelVersion) => {
        result.push({
          displayedName: modelVersion.name,
          version: modelVersion.version,
          link:
            modelVersion.name && modelVersion.version
              ? ModelRegistryRoutes.getModelVersionPageRoute(modelVersion.name, modelVersion.version)
              : '',
          status: modelVersion.status,
          source: modelVersion.source,
        });
      });
    }
    return result;
  }

  if (registeredModelsFromStore) {
    return registeredModelsFromStore.map((modelVersion) => {
      const name = modelVersion.name;
      const link = ModelRegistryRoutes.getModelVersionPageRoute(name, modelVersion.version);
      return {
        displayedName: modelVersion.name,
        version: modelVersion.version,
        link,
        status: modelVersion.status,
        source: modelVersion.source,
      };
    });
  }

  return [];
};

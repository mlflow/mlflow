import { useMemo } from 'react';
import type { LoggedModelProto } from '../../../types';
import { compact } from 'lodash';
import { RunPageModelVersionSummary } from '../../run-page/hooks/useUnifiedRegisteredModelVersionsSummariesForRun';
import { createMLflowRoutePath } from '../../../../common/utils/RoutingUtils';

const MODEL_VERSIONS_TAG_NAME = 'mlflow.modelVersions';

const isUCModelName = (name: string) => Boolean(name.match(/^[^. /]+\.[^. /]+\.[^. /]+$/));
const getUCModelUrl = (name: string, version: string) =>
  `/explore/data/models/${name.replace(/\./g, '/')}/version/${version}`;
const getWMRModelUrl = (name: string, version: string) => createMLflowRoutePath(`/models/${name}/versions/${version}`);

const getTagValueForModel = (loggedModel: LoggedModelProto): { name: string; version: string }[] | null => {
  try {
    const tagValue = loggedModel.info?.tags?.find((tag) => tag.key === MODEL_VERSIONS_TAG_NAME)?.value;

    if (tagValue) {
      // Try to parse the tag. If it's malformed, catch and return nothing.
      return JSON.parse(tagValue);
    }
  } catch (e) {
    return null;
  }
  return null;
};

export const useExperimentLoggedModelRegisteredVersions = ({ loggedModels }: { loggedModels: LoggedModelProto[] }) => {
  const parsedModelVersionsTags = useMemo<{ name: string; version: string }[]>(
    () => compact(loggedModels.map(getTagValueForModel)).flat(),
    [loggedModels],
  );

  return useMemo<RunPageModelVersionSummary[]>(
    () =>
      parsedModelVersionsTags.map((registeredModelEntry) => {
        const isUCModel = isUCModelName(registeredModelEntry.name);
        const getUrlFn = isUCModel ? getUCModelUrl : getWMRModelUrl;
        return {
          displayedName: registeredModelEntry.name,
          version: registeredModelEntry.version,
          link: getUrlFn(registeredModelEntry.name, registeredModelEntry.version),
          source: null,
          status: null,
        };
      }) ?? [],
    [parsedModelVersionsTags],
  );
};

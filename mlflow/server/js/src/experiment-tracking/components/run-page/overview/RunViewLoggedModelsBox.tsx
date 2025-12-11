import { ModelsIcon, Overflow, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../../common/utils/RoutingUtils';
import type { RunInfoEntity } from '../../../types';
import { type LoggedModelProto } from '../../../types';
import Routes from '../../../routes';
import { first } from 'lodash';
import { FormattedMessage } from 'react-intl';
import { useMemo } from 'react';
import type { UseGetRunQueryResponseRunInfo } from '../hooks/useGetRunQuery';

/**
 * Displays list of registered models in run detail overview.
 */
export const RunViewLoggedModelsBox = ({
  loggedModels,
  loggedModelsV3,
  runInfo,
}: {
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  loggedModelsV3: LoggedModelProto[];
  loggedModels: {
    artifactPath: string;
    flavors: string[];
    utcTimeCreated: number;
  }[];
}) => {
  const { theme } = useDesignSystemTheme();
  const { experimentId, runUuid } = runInfo;

  const getModelFlavorName = (flavors: string[]) => {
    return (
      first(flavors) || (
        <FormattedMessage
          defaultMessage="Model"
          description="Run page > Overview > Logged models > Unknown model flavor"
        />
      )
    );
  };

  // Check if list has models with same flavor names.
  // If true, display artifact path in dropdown menu to reduce ambiguity.
  const shouldDisplayArtifactPaths = useMemo(() => {
    const flavors = loggedModels.map((model) => getModelFlavorName(model.flavors));
    const uniqueFlavors = new Set(flavors);
    return uniqueFlavors.size !== flavors.length;
  }, [loggedModels]);

  return (
    <Overflow>
      {loggedModels.map((model, index) => {
        return (
          <Link
            to={Routes.getRunPageRoute(experimentId ?? '', runUuid ?? '', model.artifactPath)}
            key={model.artifactPath}
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              cursor: 'pointer',
              height: shouldDisplayArtifactPaths && index > 0 ? theme.general.heightBase : theme.general.heightSm,
            }}
          >
            <ModelsIcon />
            <div>
              {getModelFlavorName(model.flavors)}
              {shouldDisplayArtifactPaths && index > 0 && <Typography.Hint>{model.artifactPath}</Typography.Hint>}
            </div>
          </Link>
        );
      })}
      {loggedModelsV3.map((model, index) => {
        return (
          <Link
            to={Routes.getExperimentLoggedModelDetailsPageRoute(experimentId ?? '', model.info?.model_id ?? '')}
            key={model.info?.model_id ?? index}
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              cursor: 'pointer',
              height: shouldDisplayArtifactPaths && index > 0 ? theme.general.heightBase : theme.general.heightSm,
            }}
          >
            <ModelsIcon />
            <div>{model.info?.name}</div>
          </Link>
        );
      })}
    </Overflow>
  );
};

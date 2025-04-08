import { Overflow, Tag, useDesignSystemTheme } from '@databricks/design-system';
import { type LoggedModelProto } from '../../types';
import { useExperimentLoggedModelRegisteredVersions } from './hooks/useExperimentLoggedModelRegisteredVersions';
import { isEmpty } from 'lodash';
import { Link } from '../../../common/utils/RoutingUtils';
import { useMemo } from 'react';
import { ReactComponent as RegisteredModelOkIcon } from '../../../common/static/registered-model-grey-ok.svg';

export const ExperimentLoggedModelDetailsModelVersionsList = ({ loggedModel }: { loggedModel: LoggedModelProto }) => {
  const loggedModels = useMemo(() => [loggedModel], [loggedModel]);
  const { theme } = useDesignSystemTheme();
  const modelVersions = useExperimentLoggedModelRegisteredVersions({ loggedModels });

  if (isEmpty(modelVersions)) {
    return <>-</>;
  }

  return (
    <Overflow>
      {modelVersions?.map(({ displayedName, version, link }) => (
        <Link
          to={link}
          key={`${displayedName}-${version}`}
          css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
        >
          <RegisteredModelOkIcon /> {displayedName}{' '}
          <Tag componentId="mlflow.logged_model.details.registered_model_version_tag">v{version}</Tag>
        </Link>
      ))}
    </Overflow>
  );
};

import type { LoggedModelProto } from '../../types';
import { GraphQLExperimentRun } from '../../types';
import { Link } from '../../../common/utils/RoutingUtils';
import { useExperimentLoggedModelRegisteredVersions } from './hooks/useExperimentLoggedModelRegisteredVersions';
import { isEmpty } from 'lodash';
import React, { useMemo } from 'react';
import { Overflow, Tag, useDesignSystemTheme } from '@databricks/design-system';
import { ReactComponent as RegisteredModelOkIcon } from '../../../common/static/registered-model-grey-ok.svg';

export const ExperimentLoggedModelTableRegisteredModelsCell = ({ data }: { data: LoggedModelProto }) => {
  const { theme } = useDesignSystemTheme();

  const loggedModels = useMemo(() => [data], [data]);

  const { modelVersions } = useExperimentLoggedModelRegisteredVersions({ loggedModels });

  if (!isEmpty(modelVersions)) {
    return (
      <Overflow>
        {modelVersions.map((modelVersion) => (
          <React.Fragment key={modelVersion.link}>
            <Link to={modelVersion.link} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <RegisteredModelOkIcon />
              {modelVersion.displayedName}
              <Tag
                componentId="mlflow.logged_model.list.registered_model_cell_version_tag"
                css={{ marginRight: 0, verticalAlign: 'middle' }}
              >
                v{modelVersion.version}
              </Tag>
            </Link>
          </React.Fragment>
        ))}
      </Overflow>
    );
  }
  return '-';
};

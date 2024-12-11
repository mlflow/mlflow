import { Overflow, Tag, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../../common/utils/RoutingUtils';
import { ModelVersionInfoEntity, RunInfoEntity } from '../../../types';
import { ModelRegistryRoutes } from '../../../../model-registry/routes';
import { ReactComponent as RegisteredModelOkIcon } from '../../../../common/static/registered-model-grey-ok.svg';
import type { UseGetRunQueryResponseRunInfo } from '../hooks/useGetRunQuery';

const getModelLink = (modelVersion: ModelVersionInfoEntity) => {
  const { name, version } = modelVersion;
  return ModelRegistryRoutes.getModelVersionPageRoute(name, version);
};

/**
 * Displays list of registered models in run detail overview.
 * TODO: expand with logged models after finalizing design
 */
export const RunViewRegisteredModelsBox = ({
  registeredModels,
  runInfo,
}: {
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  registeredModels: ModelVersionInfoEntity[];
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Overflow>
      {registeredModels.map((model) => (
        <Link
          key={model.name}
          to={getModelLink(model)}
          css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
        >
          <RegisteredModelOkIcon /> {model.name}{' '}
          <Tag
            componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewregisteredmodelsbox.tsx_40"
            css={{ cursor: 'pointer' }}
          >
            v{model.version}
          </Tag>
        </Link>
      ))}
    </Overflow>
  );
};

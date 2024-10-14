import React from 'react';
import { ModelsIcon, Overflow, Tag, LegacyTooltip, useDesignSystemTheme } from '@databricks/design-system';
import Utils from '../../../../../../common/utils/Utils';
import { ModelRegistryRoutes } from '../../../../../../model-registry/routes';
import Routes from '../../../../../routes';
import { RunRowModelsInfo } from '../../../utils/experimentPage.row-types';
import { Link } from '../../../../../../common/utils/RoutingUtils';
import { ReactComponent as RegisteredModelOkIcon } from '../../../../../../common/static/registered-model-grey-ok.svg';
import { FormattedMessage } from 'react-intl';

const EMPTY_CELL_PLACEHOLDER = '-';

export interface ModelsCellRendererProps {
  value: RunRowModelsInfo;
}

/**
 * Backfill Typescript type for the value returned from Utils.mergeLoggedAndRegisteredModels
 */
interface CombinedModelType {
  registeredModelName?: string;
  isUc?: string;
  registeredModelVersion?: string;
  artifactPath?: string;
  flavors?: string[];
}

/**
 * Icon, label and link for a single model
 */
const ModelLink = ({
  model: { isUc, registeredModelName, registeredModelVersion, flavors, artifactPath } = {},
  experimentId,
  runUuid,
}: {
  model?: CombinedModelType;
  experimentId: string;
  runUuid: string;
}) => {
  const { theme } = useDesignSystemTheme();

  // Renders a model name based on whether it's a registered model or not
  const renderModelName = () => {
    const displayFullName = `${registeredModelName} v${registeredModelVersion}`;
    if (registeredModelName) {
      return (
        <LegacyTooltip title={displayFullName} placement="topLeft">
          <span css={{ verticalAlign: 'middle' }}>{registeredModelName}</span>{' '}
          <Tag
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_modelscellrenderer.tsx_49"
            css={{ marginRight: 0, verticalAlign: 'middle' }}
          >
            v{registeredModelVersion}
          </Tag>
        </LegacyTooltip>
      );
    }

    const firstFlavorName = flavors?.[0];

    return (
      firstFlavorName || (
        <FormattedMessage
          defaultMessage="Model"
          description="Experiment page > runs table > models column > default label for no specific model"
        />
      )
    );
  };

  // Renders a link to either the model registry or the run artifacts page
  const renderModelLink = () => {
    if (registeredModelName && registeredModelVersion) {
      return ModelRegistryRoutes.getModelVersionPageRoute(registeredModelName, registeredModelVersion);
    }
    return Routes.getRunPageRoute(experimentId, runUuid, artifactPath);
  };

  // Renders an icon based on whether it's a registered model or not
  const renderModelIcon = () => {
    if (registeredModelName) {
      return <RegisteredModelOkIcon css={{ color: theme.colors.actionPrimaryBackgroundDefault }} />;
    }
    return <ModelsIcon css={{ color: theme.colors.actionPrimaryBackgroundDefault }} />;
  };

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, overflow: 'hidden' }}>
      <div css={{ width: 20, display: 'flex', alignItems: 'center', justifyContent: 'flex-start', flexShrink: 0 }}>
        {renderModelIcon()}
      </div>
      <Link
        to={renderModelLink()}
        target="_blank"
        css={{ textOverflow: 'ellipsis', overflow: 'hidden', cursor: 'pointer' }}
      >
        {renderModelName()}
      </Link>
    </div>
  );
};

export const ModelsCellRenderer = React.memo((props: ModelsCellRendererProps) => {
  if (!props.value) {
    return <>{EMPTY_CELL_PLACEHOLDER}</>;
  }
  const { registeredModels, loggedModels, experimentId, runUuid } = props.value;
  const models: CombinedModelType[] = Utils.mergeLoggedAndRegisteredModels(loggedModels, registeredModels) as any[];

  if (models && models.length) {
    return (
      // <Overflow /> component does not ideally fit within ag-grid cell so we need to override its styles a bit
      <div css={{ width: '100%', '&>div': { maxWidth: '100%', display: 'flex' } }}>
        <Overflow>
          {models.map((model, index) => (
            <ModelLink model={model} key={model.artifactPath || index} experimentId={experimentId} runUuid={runUuid} />
          ))}
        </Overflow>
      </div>
    );
  }
  return <>{EMPTY_CELL_PLACEHOLDER}</>;
});

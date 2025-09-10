import React, { useMemo } from 'react';
import { ModelsIcon, Overflow, Tag, LegacyTooltip, useDesignSystemTheme } from '@databricks/design-system';
import Utils from '../../../../../../common/utils/Utils';
import { ModelRegistryRoutes } from '../../../../../../model-registry/routes';
import Routes from '../../../../../routes';
import type { RunRowModelsInfo } from '../../../utils/experimentPage.row-types';
import { Link } from '../../../../../../common/utils/RoutingUtils';
import { ReactComponent as RegisteredModelOkIcon } from '../../../../../../common/static/registered-model-grey-ok.svg';
import type { LoggedModelProto } from '../../../../../types';
import { FormattedMessage } from 'react-intl';
import { useExperimentLoggedModelRegisteredVersions } from '../../../../experiment-logged-models/hooks/useExperimentLoggedModelRegisteredVersions';
import { isEmpty, uniqBy, values } from 'lodash';
import { isUCModelName } from '../../../../../utils/IsUCModelName';
import {
  shouldUnifyLoggedModelsAndRegisteredModels,
  shouldUseGetLoggedModelsBatchAPI,
} from '../../../../../../common/utils/FeatureUtils';

const EMPTY_CELL_PLACEHOLDER = '-';

export interface ModelsCellRendererProps {
  value: RunRowModelsInfo;
}

/**
 * Backfill Typescript type for the value returned from Utils.mergeLoggedAndRegisteredModels
 */
interface CombinedModelType {
  registeredModelName?: string;
  isUc?: boolean;
  registeredModelVersion?: string;
  artifactPath?: string;
  flavors?: string[];
  originalLoggedModel?: LoggedModelProto;
}

/**
 * Icon, label and link for a single model
 */
const ModelLink = ({
  model: { isUc, registeredModelName, registeredModelVersion, flavors, artifactPath, originalLoggedModel } = {},
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
    let tooltipBody: React.ReactNode = `${registeredModelName} v${registeredModelVersion}`;

    // If the model is a registered model coming from V3 logged model, we need to show the original logged model name
    if (
      registeredModelName &&
      registeredModelVersion &&
      originalLoggedModel &&
      shouldUnifyLoggedModelsAndRegisteredModels()
    ) {
      const loggedModelExperimentId = originalLoggedModel.info?.experiment_id;
      const loggedModelId = originalLoggedModel.info?.model_id;
      if (loggedModelExperimentId && loggedModelId) {
        tooltipBody = (
          <FormattedMessage
            defaultMessage="Original logged model: {originalModelLink}"
            description="Tooltip text with link to the original logged model"
            values={{
              originalModelLink: (
                <Link
                  to={Routes.getExperimentLoggedModelDetailsPage(loggedModelExperimentId, loggedModelId)}
                  css={{ color: 'inherit', textDecoration: 'underline' }}
                >
                  {originalLoggedModel.info?.name}
                </Link>
              ),
            }}
          />
        );
      }
    }
    if (registeredModelName) {
      return (
        <LegacyTooltip title={tooltipBody} placement="topLeft">
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

const LoggedModelV3Link = ({ model }: { model: LoggedModelProto }) => {
  const { theme } = useDesignSystemTheme();

  if (!model.info?.model_id || !model.info?.experiment_id) {
    return null;
  }
  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, overflow: 'hidden' }}>
      <div css={{ width: 20, display: 'flex', alignItems: 'center', justifyContent: 'flex-start', flexShrink: 0 }}>
        <ModelsIcon css={{ color: theme.colors.actionPrimaryBackgroundDefault }} />
      </div>
      <Link
        to={Routes.getExperimentLoggedModelDetailsPage(model.info.experiment_id, model.info.model_id)}
        target="_blank"
        css={{ textOverflow: 'ellipsis', overflow: 'hidden', cursor: 'pointer' }}
      >
        {model.info.name}
      </Link>
    </div>
  );
};

/**
 * This component renders combined set of models, based on provided models payload.
 * The models are sourced from:
 * - `registeredModels` containing WMR and UC model versions associated with the run, populated by API call
 * - `loggedModels` containing legacy (pre-V3) logged models associated with the run, listed in run's tag
 * - `loggedModelsV3` containing V3 logged models associated with the runs inputs and outputs, populated by API call
 * In the component, we also resolve registered model versions for V3 logged models based on loged model's tags
 */
export const ModelsCellRenderer = React.memo((props: ModelsCellRendererProps) => {
  const { registeredModels = [], loggedModels = [], loggedModelsV3, experimentId, runUuid } = props.value || {};

  // First, we merge legacy logged models and registered models.
  const modelsLegacy: CombinedModelType[] = Utils.mergeLoggedAndRegisteredModels(
    loggedModels,
    registeredModels,
  ) as any[];

  // Next, registered model versions are resolved from V3 logged models' tags
  const { modelVersions: registeredModelVersions } = useExperimentLoggedModelRegisteredVersions({
    loggedModels: loggedModelsV3 || [],
  });

  // We create a map of registered model versions by their source logged model.
  // This allows to unfurl logged model to registered model versions while hiding the original logged model.
  const registeredModelVersionsByLoggedModel = useMemo(() => {
    if (!shouldUseGetLoggedModelsBatchAPI()) {
      return {};
    }
    const map: Record<string, CombinedModelType[]> = {};
    registeredModelVersions.forEach((modelVersion) => {
      const loggedModelId = modelVersion.sourceLoggedModel?.info?.model_id;
      if (loggedModelId) {
        const registeredModels = map[loggedModelId] || [];
        const name = modelVersion.displayedName ?? undefined;
        registeredModels.push({
          registeredModelName: name,
          registeredModelVersion: modelVersion.version ?? undefined,
          isUc: isUCModelName(name ?? ''),
          artifactPath: modelVersion.sourceLoggedModel?.info?.artifact_uri ?? '',
          flavors: [],
          originalLoggedModel: modelVersion.sourceLoggedModel,
        });
        map[loggedModelId] = registeredModels;
      }
    });
    return map;
  }, [registeredModelVersions]);

  // Merge legacy models with registered model versions from V3 logged models.
  const registeredModelsToDisplay = useMemo(() => {
    const allModels = [...modelsLegacy, ...Array.from(values(registeredModelVersionsByLoggedModel)).flat()];
    // Remove duplicates (it's not impossible to reference the same model version twice in a single logged model)
    return uniqBy(allModels, (model) =>
      JSON.stringify(
        model.registeredModelName && model.registeredModelVersion
          ? [model.registeredModelName, model.registeredModelVersion?.toString()]
          : [model.artifactPath],
      ),
    );
  }, [modelsLegacy, registeredModelVersionsByLoggedModel]);

  const containsModels = !isEmpty(registeredModelsToDisplay) || !isEmpty(loggedModelsV3);

  if (!props.value) {
    return <>{EMPTY_CELL_PLACEHOLDER}</>;
  }

  if (containsModels) {
    return (
      // <Overflow /> component does not ideally fit within ag-grid cell so we need to override its styles a bit
      <div css={{ width: '100%', '&>div': { maxWidth: '100%', display: 'flex' } }}>
        <Overflow>
          {registeredModelsToDisplay.map((model, index) => (
            <ModelLink model={model} key={model.artifactPath || index} experimentId={experimentId} runUuid={runUuid} />
          ))}
          {loggedModelsV3?.map((model, index) => {
            // Display logged model only if it does not have registered model versions associated with it.
            const modelId = model.info?.model_id;
            const loggedModelRegisteredVersions = modelId ? registeredModelVersionsByLoggedModel[modelId] : [];
            if (!isEmpty(loggedModelRegisteredVersions)) {
              return null;
            }

            return <LoggedModelV3Link key={model.info?.model_id ?? index} model={model} />;
          })}
        </Overflow>
      </div>
    );
  }
  return <>{EMPTY_CELL_PLACEHOLDER}</>;
});

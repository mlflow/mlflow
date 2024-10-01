import {
  Button,
  ChevronDownIcon,
  DropdownMenu,
  NewWindowIcon,
  Tag,
  LegacyTooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { first, last, orderBy } from 'lodash';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useSelector } from 'react-redux';
import { Link } from '../../../common/utils/RoutingUtils';
import Utils from '../../../common/utils/Utils';
import { RegisterModel } from '../../../model-registry/components/RegisterModel';
import { ModelVersionStatusIcons } from '../../../model-registry/constants';
import { ModelRegistryRoutes } from '../../../model-registry/routes';
import type { ReduxState } from '../../../redux-types';
import Routes from '../../routes';
import { KeyValueEntity, ModelVersionInfoEntity } from '../../types';
import { ReactComponent as RegisteredModelOkIcon } from '../../../common/static/registered-model-grey-ok.svg';

interface LoggedModelWithRegistrationInfo {
  path: string;
  absolutePath: string;
  registeredVersions: ModelVersionInfoEntity[];
}

export function LoggedModelsDropdownContent({
  models,
  onRegisterClick,
  experimentId,
  runUuid,
}: {
  models: LoggedModelWithRegistrationInfo[];
  onRegisterClick: (model: LoggedModelWithRegistrationInfo) => void;
  experimentId: string;
  runUuid: string;
}) {
  const { theme } = useDesignSystemTheme();
  const renderSection = (title: string, sectionModels: LoggedModelWithRegistrationInfo[]) => {
    return (
      <DropdownMenu.Group>
        <DropdownMenu.Label>{title}</DropdownMenu.Label>
        {sectionModels.map((model) => {
          const registeredModel = first(model.registeredVersions);
          if (!registeredModel) {
            return (
              <DropdownMenu.Item
                componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_50"
                onClick={() => onRegisterClick(model)}
                key={model.absolutePath}
              >
                <div css={{ marginRight: theme.spacing.md }}>{last(model.path.split('/'))}</div>
                <DropdownMenu.HintColumn>
                  <Link
                    target="_blank"
                    to={Routes.getRunPageTabRoute(experimentId, runUuid, 'artifacts/' + model.path)}
                  >
                    <Button
                      componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_58"
                      type="link"
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                      }}
                      endIcon={<NewWindowIcon />}
                    >
                      <FormattedMessage
                        defaultMessage="View model"
                        description="Run page > Header > Register model dropdown > View model button label"
                      />
                    </Button>
                  </Link>
                </DropdownMenu.HintColumn>
              </DropdownMenu.Item>
            );
          }
          const { status, name, version } = registeredModel;

          return (
            <Link target="_blank" to={getRegisteredModelVersionLink(registeredModel)} key={model.absolutePath}>
              <DropdownMenu.Item componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_80">
                <DropdownMenu.IconWrapper css={{ display: 'flex', alignItems: 'center' }}>
                  {status === 'READY' ? <RegisteredModelOkIcon /> : ModelVersionStatusIcons[status]}
                </DropdownMenu.IconWrapper>
                <span css={{ marginRight: theme.spacing.md }}>
                  {name}
                  <Tag
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_90"
                    css={{ marginLeft: 8, marginRight: 4 }}
                  >
                    v{version}
                  </Tag>
                </span>
                <DropdownMenu.HintColumn>
                  <Button
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_89"
                    type="link"
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                    }}
                    endIcon={<NewWindowIcon />}
                  >
                    <FormattedMessage
                      defaultMessage="Go to model"
                      description="Run page > Header > Register model dropdown > Go to model button label"
                    />
                  </Button>
                </DropdownMenu.HintColumn>
              </DropdownMenu.Item>
            </Link>
          );
        })}
      </DropdownMenu.Group>
    );
  };
  const registeredModels = models.filter((model) => model.registeredVersions.length > 0);
  const unregisteredModels = models.filter((model) => !model.registeredVersions.length);
  return (
    <>
      {unregisteredModels.length ? renderSection('Unregistered models', unregisteredModels) : null}
      {unregisteredModels.length && registeredModels.length ? <DropdownMenu.Separator /> : null}
      {registeredModels.length ? renderSection('Registered models', registeredModels) : null}
    </>
  );
}

const getRegisteredModelVersionLink = (modelVersion: ModelVersionInfoEntity) => {
  const { name, version } = modelVersion;
  return ModelRegistryRoutes.getModelVersionPageRoute(name, version);
};

export const RunViewHeaderRegisterModelButton = ({
  runUuid,
  experimentId,
  runTags,
  artifactRootUri,
}: {
  runUuid: string;
  experimentId: string;
  runTags: Record<string, KeyValueEntity>;
  artifactRootUri?: string;
}) => {
  const { theme } = useDesignSystemTheme();

  const registeredModelVersions = useSelector((state: ReduxState) =>
    orderBy(state.entities.modelVersionsByRunUuid[runUuid]),
  );
  const loggedModelPaths = useMemo(
    () => (runTags ? Utils.getLoggedModelsFromTags(runTags).map(({ artifactPath }) => artifactPath) : []),
    [runTags],
  );

  const models = useMemo<LoggedModelWithRegistrationInfo[]>(
    () =>
      orderBy(
        loggedModelPaths.map((path) => ({
          path,
          absolutePath: `${artifactRootUri}/${path}`,
          registeredVersions:
            registeredModelVersions?.filter(({ source }) => source === `${artifactRootUri}/${path}`) || [],
        })),
        (model) => parseInt(model.registeredVersions[0]?.version || '0', 10),
        'desc',
      ),
    [loggedModelPaths, registeredModelVersions, artifactRootUri],
  );

  const [selectedModelToRegister, setSelectedModelToRegister] = useState<LoggedModelWithRegistrationInfo | null>(null);

  if (models.length > 1) {
    const modelsRegistered = models.filter((model) => model.registeredVersions.length > 0);

    return (
      <>
        {selectedModelToRegister && (
          <RegisterModel
            runUuid={runUuid}
            modelPath={selectedModelToRegister.absolutePath}
            modelRelativePath={selectedModelToRegister.path}
            disabled={false}
            showButton={false}
            modalVisible
            onCloseModal={() => setSelectedModelToRegister(null)}
          />
        )}
        <DropdownMenu.Root modal={false}>
          <LegacyTooltip
            placement="bottom"
            title={
              <FormattedMessage
                defaultMessage="{registeredCount}/{loggedCount} logged models are registered"
                description="Run page > Header > Register model dropdown > Button tooltip"
                values={{ registeredCount: modelsRegistered.length, loggedCount: models.length }}
              />
            }
          >
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_195"
                type="primary"
                endIcon={<ChevronDownIcon />}
              >
                <FormattedMessage
                  defaultMessage="Register model"
                  description="Run page > Header > Register model dropdown > Button label when some models are not registered"
                />
              </Button>
            </DropdownMenu.Trigger>
          </LegacyTooltip>
          <DropdownMenu.Content align="end">
            <LoggedModelsDropdownContent
              models={models}
              onRegisterClick={setSelectedModelToRegister}
              experimentId={experimentId}
              runUuid={runUuid}
            />
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </>
    );
  }

  const singleModel = first(models);

  if (!singleModel) {
    return null;
  }

  const registeredModelVersion = first(singleModel.registeredVersions);

  if (registeredModelVersion) {
    return (
      <Link
        to={getRegisteredModelVersionLink(registeredModelVersion)}
        target="_blank"
        css={{ marginLeft: theme.spacing.sm }}
      >
        <Button
          componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_231"
          endIcon={<NewWindowIcon />}
          type="link"
        >
          Model registered
        </Button>
      </Link>
    );
  }
  return (
    <RegisterModel
      disabled={false}
      runUuid={runUuid}
      modelPath={singleModel.absolutePath}
      modelRelativePath={singleModel.path}
      showButton
      buttonType="primary"
    />
  );
};

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
import Routes from '../../routes';
import type { ModelVersionInfoEntity } from '../../types';
import type { KeyValueEntity } from '../../../common/types';
import { ReactComponent as RegisteredModelOkIcon } from '../../../common/static/registered-model-grey-ok.svg';
import type { RunPageModelVersionSummary } from './hooks/useUnifiedRegisteredModelVersionsSummariesForRun';

interface LoggedModelWithRegistrationInfo {
  path: string;
  absolutePath: string;
  registeredModelVersionSummaries: RunPageModelVersionSummary[];
}

function LoggedModelsDropdownContent({
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
          const registeredModelSummary = first(model.registeredModelVersionSummaries);
          if (!registeredModelSummary) {
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
          const { status, displayedName, version, link } = registeredModelSummary;

          return (
            <Link target="_blank" to={link} key={model.absolutePath}>
              <DropdownMenu.Item componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewheaderregistermodelbutton.tsx_80">
                <DropdownMenu.IconWrapper css={{ display: 'flex', alignItems: 'center' }}>
                  {status === 'READY' ? <RegisteredModelOkIcon /> : status ? ModelVersionStatusIcons[status] : null}
                </DropdownMenu.IconWrapper>
                <span css={{ marginRight: theme.spacing.md }}>
                  {displayedName}
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
  const registeredModels = models.filter((model) => model.registeredModelVersionSummaries.length > 0);
  const unregisteredModels = models.filter((model) => !model.registeredModelVersionSummaries.length);
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
  registeredModelVersionSummaries,
}: {
  runUuid: string;
  experimentId: string;
  runTags: Record<string, KeyValueEntity>;
  artifactRootUri?: string;
  registeredModelVersionSummaries: RunPageModelVersionSummary[];
}) => {
  const { theme } = useDesignSystemTheme();

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
          registeredModelVersionSummaries:
            registeredModelVersionSummaries?.filter(({ source }) => source === `${artifactRootUri}/${path}`) || [],
        })),
        (model) => parseInt(model.registeredModelVersionSummaries[0]?.version || '0', 10),
        'desc',
      ),
    [loggedModelPaths, registeredModelVersionSummaries, artifactRootUri],
  );

  const [selectedModelToRegister, setSelectedModelToRegister] = useState<LoggedModelWithRegistrationInfo | null>(null);

  if (models.length > 1) {
    const modelsRegistered = models.filter((model) => model.registeredModelVersionSummaries.length > 0);

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

  const registeredModelVersionSummary = first(singleModel.registeredModelVersionSummaries);

  if (registeredModelVersionSummary) {
    return (
      <Link to={registeredModelVersionSummary.link} target="_blank" css={{ marginLeft: theme.spacing.sm }}>
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

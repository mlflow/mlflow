import { Button, ChevronDownIcon, DropdownMenu, NewWindowIcon, useDesignSystemTheme } from '@databricks/design-system';
import { first } from 'lodash';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { RegisterModel } from '../../../model-registry/components/RegisterModel';
import Routes from '../../routes';
import type { LoggedModelProto } from '../../types';

/**
 * A specialized variant of the "Register model" button intended for the run page,
 * which uses logged models from IAv3.
 */
export const RunViewHeaderRegisterFromLoggedModelV3Button = ({
  runUuid,
  experimentId,
  loggedModels,
}: {
  runUuid: string;
  experimentId: string;
  loggedModels: LoggedModelProto[];
}) => {
  const { theme } = useDesignSystemTheme();

  const [selectedModelToRegister, setSelectedModelToRegister] = useState<LoggedModelProto | null>(null);

  const singleModel = first(loggedModels);

  if (loggedModels.length > 1) {
    return (
      <>
        {selectedModelToRegister?.info?.artifact_uri && (
          <RegisterModel
            runUuid={runUuid}
            modelPath={selectedModelToRegister.info.artifact_uri}
            loggedModelId={selectedModelToRegister.info.model_id}
            modelRelativePath=""
            disabled={false}
            showButton={false}
            modalVisible
            onCloseModal={() => setSelectedModelToRegister(null)}
          />
        )}
        <DropdownMenu.Root modal={false}>
          <DropdownMenu.Trigger asChild>
            <Button
              componentId="mlflow.run_details.header.register_model_from_logged_model.button"
              type="primary"
              endIcon={<ChevronDownIcon />}
            >
              <FormattedMessage
                defaultMessage="Register model"
                description="Label for a CTA button for registering a ML model version from a logged model"
              />
            </Button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align="end">
            {loggedModels.map((model) => {
              return (
                <DropdownMenu.Item
                  componentId="mlflow.run_details.header.register_model_from_logged_model.dropdown_menu_item"
                  onClick={() => setSelectedModelToRegister(model)}
                  key={model.info?.model_id}
                >
                  <div css={{ marginRight: theme.spacing.md }}>{model.info?.name}</div>
                  <DropdownMenu.HintColumn>
                    <Link
                      target="_blank"
                      to={Routes.getExperimentLoggedModelDetailsPage(experimentId, model.info?.model_id ?? '')}
                    >
                      <Button
                        componentId="mlflow.run_details.header.register_model_from_logged_model.dropdown_menu_item.view_model_button"
                        type="link"
                        size="small"
                        onClick={(e) => e.stopPropagation()}
                        endIcon={<NewWindowIcon />}
                      >
                        <FormattedMessage
                          defaultMessage="View model"
                          description="Label for a button that opens a new tab to view the details of a logged ML model while registering a model version"
                        />
                      </Button>
                    </Link>
                  </DropdownMenu.HintColumn>
                </DropdownMenu.Item>
              );
            })}
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </>
    );
  }

  if (!singleModel || !singleModel.info?.artifact_uri) {
    return null;
  }

  return (
    <RegisterModel
      disabled={false}
      runUuid={runUuid}
      modelPath={singleModel.info.artifact_uri}
      loggedModelId={singleModel.info.model_id}
      modelRelativePath=""
      showButton
      buttonType="primary"
    />
  );
};

import { FormattedMessage, useIntl } from 'react-intl';

import { FormUI, Input, Spacer } from '@databricks/design-system';
import { shouldEnableWorkspaces } from '../../../common/utils/FeatureUtils';

type CreateExperimentFormProps = {
  experimentName: string;
  artifactLocation: string;
  nameError: string;
  onNameChange: (value: string) => void;
  onArtifactLocationChange: (value: string) => void;
};

export const CreateExperimentForm = ({
  experimentName,
  artifactLocation,
  nameError,
  onNameChange,
  onArtifactLocationChange,
}: CreateExperimentFormProps) => {
  const intl = useIntl();
  const workspacesEnabled = shouldEnableWorkspaces();

  return (
    <>
      <FormUI.Label htmlFor="mlflow.create_experiment.name">
        <FormattedMessage
          defaultMessage="Experiment Name"
          description="Label for create experiment modal to enter a valid experiment name"
        />
      </FormUI.Label>
      <Input
        componentId="mlflow.create_experiment.name_input"
        id="mlflow.create_experiment.name"
        value={experimentName}
        onChange={(e) => onNameChange(e.target.value)}
        placeholder={intl.formatMessage({
          defaultMessage: 'Input an experiment name',
          description: 'Input placeholder to enter experiment name for create experiment',
        })}
        validationState={nameError ? 'error' : undefined}
        autoFocus
      />
      {nameError && <FormUI.Message type="error" message={nameError} />}
      {!workspacesEnabled && (
        <>
          <Spacer />
          <FormUI.Label htmlFor="mlflow.create_experiment.artifact_location">
            <FormattedMessage
              defaultMessage="Artifact Location"
              description="Label for create experiment modal to enter a artifact location"
            />
          </FormUI.Label>
          <Input
            componentId="mlflow.create_experiment.artifact_location_input"
            id="mlflow.create_experiment.artifact_location"
            value={artifactLocation}
            onChange={(e) => onArtifactLocationChange(e.target.value)}
            placeholder={intl.formatMessage({
              defaultMessage: 'Input an artifact location (optional)',
              description: 'Input placeholder to enter artifact location for create experiment',
            })}
          />
        </>
      )}
    </>
  );
};

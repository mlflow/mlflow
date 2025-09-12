import { FormattedMessage, useIntl } from 'react-intl';
import type { LoggedModelProto } from '../../types';
import { RegisterModel } from '../../../model-registry/components/RegisterModel';
import { useCallback } from 'react';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '../../../shared/building_blocks/CopyButton';
import { CopyIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useValidateLoggedModelSignature } from './hooks/useValidateLoggedModelSignature';
import Utils from '../../../common/utils/Utils';

const RegisterLoggedModelInUCCodeSnippet = ({ modelId }: { modelId: string }) => {
  const { theme } = useDesignSystemTheme();

  const code = `import mlflow

mlflow.set_registry_uri("databricks-uc")

model_uri = "models:/${modelId}"
model_name = "main.default.my_model"

mlflow.register_model(model_uri=model_uri, name=model_name)
`;

  return (
    <div>
      <Typography.Text>
        <FormattedMessage
          defaultMessage="In order to register model in Unity Catalog, copy and run the following code in the notebook:"
          description="Instruction to register model in Unity Catalog on the logged model details page"
        />
      </Typography.Text>
      <div css={{ position: 'relative' }}>
        <CopyButton
          css={{ zIndex: 1, position: 'absolute', top: theme.spacing.sm, right: theme.spacing.sm }}
          showLabel={false}
          copyText={code}
          icon={<CopyIcon />}
        />
        <CodeSnippet
          showLineNumbers
          style={{
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
            marginTop: theme.spacing.md,
            marginBottom: theme.spacing.md,
          }}
          language="python"
        >
          {code}
        </CodeSnippet>
      </div>
    </div>
  );
};

export const ExperimentLoggedModelDetailsRegisterButton = ({
  loggedModel,
  onSuccess,
}: {
  loggedModel?: LoggedModelProto | null;
  onSuccess?: () => void;
}) => {
  const intl = useIntl();
  const handleSuccess = useCallback(
    (data?: { value: { status?: string } }) => {
      onSuccess?.();
      const successTitle = intl.formatMessage({
        defaultMessage: 'Model registered successfully',
        description: 'Notification title for model registration succeeded on the logged model details page',
      });
      Utils.displayGlobalInfoNotification(`${successTitle} ${data?.value?.status ?? ''}`);
    },
    [intl, onSuccess],
  );

  const handleError = useCallback(
    (error?: Error | ErrorWrapper) => {
      const errorTitle = intl.formatMessage({
        defaultMessage: 'Error registering model',
        description: 'Notification title for model registration failure on the logged model details page',
      });
      const message = (error instanceof ErrorWrapper ? error.getMessageField() : error?.message) ?? String(error);
      Utils.displayGlobalErrorNotification(`${errorTitle} ${message}`);
    },
    [intl],
  );

  /**
   * Function that validates that the model file is valid to be registered in UC (contains signature inputs and outputs),
   * passed to the RegisterModel component.
   */
  const modelFileValidationFn = useValidateLoggedModelSignature(loggedModel);

  if (!loggedModel?.info?.artifact_uri || !loggedModel.info.model_id) {
    return null;
  }

  return (
    <RegisterModel
      modelPath={loggedModel.info.artifact_uri}
      modelRelativePath=""
      disabled={false}
      loggedModelId={loggedModel.info.model_id}
      buttonType="primary"
      showButton
      onRegisterSuccess={handleSuccess}
      onRegisterFailure={handleError}
    />
  );
};

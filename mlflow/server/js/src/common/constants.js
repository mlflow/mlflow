import { FormattedMessage } from 'react-intl';
import React from 'react';

export const ErrorCodes = {
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  INVALID_PARAMETER_VALUE: 'INVALID_PARAMETER_VALUE',
  RESOURCE_DOES_NOT_EXIST: 'RESOURCE_DOES_NOT_EXIST',
  PERMISSION_DENIED: 'PERMISSION_DENIED',
};

export const HomePageDocsUrl = 'https://www.mlflow.org/docs/latest/index.html';

export const ModelRegistryDocUrl = 'https://mlflow.org/docs/latest/model-registry.html';

export const ModelRegistryOnboardingString = (
  <FormattedMessage
    defaultMessage='Share and manage machine learning models.'
    description='Default text for model registry onboarding on the model list page'
  />
);

export const RegisteringModelDocUrl =
  'https://mlflow.org/docs/latest/' +
  'model-registry.html#adding-an-mlflow-model-to-the-model-registry';

export const ExperimentTrackingDocUrl = 'https://www.mlflow.org/docs/latest/tracking.html';

export const PyfuncDocUrl = 'https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html';
export const CustomPyfuncModelsDocUrl =
  'https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models';

export const LoggingRunsDocUrl =
  'https://www.mlflow.org/docs/latest/tracking.html#logging-data-to-runs';

export const onboarding = 'onboarding';

export const SupportPageUrl = 'https://github.com/mlflow/mlflow/issues';

export const ModelSignatureUrl = 'https://mlflow.org/docs/latest/models.html#model-signature';

export const LogModelWithSignatureUrl =
  'https://www.mlflow.org/docs/latest/models.html#how-to-log-models-with-signatures';

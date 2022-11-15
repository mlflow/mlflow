import { FormattedMessage } from 'react-intl';
import React from 'react';

export const ErrorCodes = {
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  INVALID_PARAMETER_VALUE: 'INVALID_PARAMETER_VALUE',
  RESOURCE_DOES_NOT_EXIST: 'RESOURCE_DOES_NOT_EXIST',
  PERMISSION_DENIED: 'PERMISSION_DENIED',
};

export const Version = '2.0.1';

const DOCS_VERSION = 'latest';

const DOCS_ROOT = `https://www.mlflow.org/docs/${DOCS_VERSION}`;

export const HomePageDocsUrl = `${DOCS_ROOT}/index.html`;

export const ModelRegistryDocUrl = `${DOCS_ROOT}/model-registry.html`;

export const ModelRegistryOnboardingString = (
  <FormattedMessage
    defaultMessage='Share and manage machine learning models.'
    description='Default text for model registry onboarding on the model list page'
  />
);

export const RegisteringModelDocUrl =
  DOCS_ROOT + '/model-registry.html#adding-an-mlflow-model-to-the-model-registry';

export const ExperimentCliDocUrl = `${DOCS_ROOT}/cli.html#mlflow-experiments`;

export const ExperimentSearchSyntaxDocUrl = `${DOCS_ROOT}/search-runs.html`;

export const ExperimentTrackingDocUrl = `${DOCS_ROOT}/tracking.html`;

export const PyfuncDocUrl = `${DOCS_ROOT}/python_api/mlflow.pyfunc.html`;
export const CustomPyfuncModelsDocUrl =
  DOCS_ROOT + '/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models';

export const LoggingRunsDocUrl = `${DOCS_ROOT}/tracking.html#logging-data-to-runs`;

export const onboarding = 'onboarding';

export const SupportPageUrl = 'https://github.com/mlflow/mlflow/issues';

export const ModelSignatureUrl = `${DOCS_ROOT}/models.html#model-signature`;

export const LogModelWithSignatureUrl =
  DOCS_ROOT + '/models.html#how-to-log-models-with-signatures';

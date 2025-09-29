import { ExperimentKind } from '../constants';
import type { MessageDescriptor } from 'react-intl';
import { defineMessage } from 'react-intl';
import type { KeyValueEntity } from '../../common/types';

export const EXPERIMENT_KIND_TAG_KEY = 'mlflow.experimentKind';

export const getExperimentKindFromTags = (
  experimentTags?:
    | ({ __typename: 'MlflowExperimentTag'; key: string | null; value: string | null }[] | null)
    | KeyValueEntity[],
): ExperimentKind | undefined =>
  experimentTags?.find((tag) => tag.key === EXPERIMENT_KIND_TAG_KEY)?.value as ExperimentKind;

export const isEditableExperimentKind = (experimentKind: ExperimentKind): boolean =>
  experimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED ||
  experimentKind === ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED ||
  experimentKind === ExperimentKind.NO_INFERRED_TYPE ||
  experimentKind === ExperimentKind.GENAI_DEVELOPMENT ||
  experimentKind === ExperimentKind.CUSTOM_MODEL_DEVELOPMENT ||
  experimentKind === ExperimentKind.EMPTY;

export const normalizeInferredExperimentKind = (experimentKind: ExperimentKind): ExperimentKind => {
  if (experimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED) {
    return ExperimentKind.GENAI_DEVELOPMENT;
  }
  if (experimentKind === ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED) {
    return ExperimentKind.CUSTOM_MODEL_DEVELOPMENT;
  }
  return experimentKind;
};

export const ExperimentKindDropdownLabels: Record<ExperimentKind, MessageDescriptor> = {
  [ExperimentKind.GENAI_DEVELOPMENT]: defineMessage({
    defaultMessage: 'GenAI apps & agents',
    description: 'Label for experiments focused on generative AI model development',
  }),
  [ExperimentKind.CUSTOM_MODEL_DEVELOPMENT]: defineMessage({
    defaultMessage: 'Machine learning',
    description: 'Label for custom experiments focused on machine learning',
  }),
  [ExperimentKind.GENAI_DEVELOPMENT_INFERRED]: defineMessage({
    defaultMessage: 'GenAI apps & agents',
    description: 'Label for experiments automatically identified as generative AI development',
  }),
  [ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED]: defineMessage({
    defaultMessage: 'Machine learning',
    description: 'Label for custom experiments automatically identified as being focused on machine learning',
  }),
  [ExperimentKind.NO_INFERRED_TYPE]: defineMessage({
    defaultMessage: 'None',
    description: 'Label for experiments with no automatically inferred experiment type',
  }),
  [ExperimentKind.FINETUNING]: defineMessage({
    defaultMessage: 'Finetuning',
    description: 'Label for experiments focused on model finetuning',
  }),
  [ExperimentKind.REGRESSION]: defineMessage({
    defaultMessage: 'Regression',
    description: 'Label for experiments focused on regression modeling',
  }),
  [ExperimentKind.CLASSIFICATION]: defineMessage({
    defaultMessage: 'Classification',
    description: 'Label for experiments focused on classification modeling',
  }),
  [ExperimentKind.FORECASTING]: defineMessage({
    defaultMessage: 'Forecasting',
    description: 'Label for experiments focused on time series forecasting',
  }),
  [ExperimentKind.AUTOML]: defineMessage({
    defaultMessage: 'AutoML',
    description: 'Label for generic AutoML experiments',
  }),
  [ExperimentKind.EMPTY]: defineMessage({
    defaultMessage: 'None',
    description: 'Label for experiments with no experiment kind',
  }),
};

export const ExperimentKindShortLabels: Record<ExperimentKind, MessageDescriptor> = {
  [ExperimentKind.GENAI_DEVELOPMENT]: defineMessage({
    defaultMessage: 'GenAI apps & agents',
    description: 'A short label for custom experiments focused on generative AI app and agent development',
  }),
  [ExperimentKind.CUSTOM_MODEL_DEVELOPMENT]: defineMessage({
    defaultMessage: 'Machine learning',
    description: 'A short label for custom experiments focused on machine learning',
  }),
  [ExperimentKind.GENAI_DEVELOPMENT_INFERRED]: defineMessage({
    defaultMessage: 'GenAI apps & agents',
    description:
      'A short label for custom experiments automatically identified as being focused on generative AI app and agent development',
  }),
  [ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED]: defineMessage({
    defaultMessage: 'Machine learning',
    description: 'A short label for custom experiments automatically identified as being focused on machine learning',
  }),
  [ExperimentKind.NO_INFERRED_TYPE]: defineMessage({
    defaultMessage: 'None',
    description: 'A short label for experiments with no automatically inferred experiment type',
  }),
  [ExperimentKind.FINETUNING]: defineMessage({
    defaultMessage: 'finetuning',
    description: 'A short label for experiments focused on model finetuning',
  }),
  [ExperimentKind.REGRESSION]: defineMessage({
    defaultMessage: 'regression',
    description: 'A short label for experiments focused on regression modeling',
  }),
  [ExperimentKind.CLASSIFICATION]: defineMessage({
    defaultMessage: 'classification',
    description: 'A short label for experiments focused on classification modeling',
  }),
  [ExperimentKind.FORECASTING]: defineMessage({
    defaultMessage: 'forecasting',
    description: 'A short label for experiments focused on time series forecasting',
  }),
  [ExperimentKind.AUTOML]: defineMessage({
    defaultMessage: 'AutoML',
    description: 'A short label for generic AutoML experiments',
  }),
  [ExperimentKind.EMPTY]: defineMessage({
    defaultMessage: 'None',
    description: 'A short label for experiments with no experiment kind',
  }),
};

// Returns list of experiment kinds that are user-selectable in the dropdown
export const getSelectableExperimentKinds = () => [
  ExperimentKind.GENAI_DEVELOPMENT,
  ExperimentKind.CUSTOM_MODEL_DEVELOPMENT,
];

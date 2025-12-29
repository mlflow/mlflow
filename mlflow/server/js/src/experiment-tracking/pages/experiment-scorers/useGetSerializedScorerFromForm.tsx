import { useFormContext } from 'react-hook-form';
import { DEFAULT_LLM_MODEL, SCORER_TYPE } from './constants';
import {
  convertFormDataToScheduledScorer,
  ScorerFormData,
  transformScheduledScorer,
} from './utils/scorerTransformUtils';
import { useCallback } from 'react';

/**
 * Returns a function that can be used to build the serialized scorer based on the current form data.
 */
export const useGetSerializedScorerFromForm = () => {
  const { getValues } = useFormContext<ScorerFormData>();

  return useCallback(() => {
    const formData = getValues();
    if (formData.scorerType === SCORER_TYPE.LLM) {
      // Fill in the default model if not set
      formData.model = formData.model || DEFAULT_LLM_MODEL;
    }
    // Convert the form data to a scheduled scorer
    const scheduledScorer = convertFormDataToScheduledScorer(formData, undefined);
    // Transform the scheduled scorer to a backend scorer config
    const scorerConfig = transformScheduledScorer(scheduledScorer);
    // Return the serialized scorer
    return scorerConfig.serialized_scorer;
  }, [getValues]);
};

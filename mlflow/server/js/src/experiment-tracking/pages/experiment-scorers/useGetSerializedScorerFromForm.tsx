import { useFormContext } from 'react-hook-form';
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
    // Convert the form data to a scheduled scorer
    const scheduledScorer = convertFormDataToScheduledScorer(formData, undefined);
    // Transform the scheduled scorer to a backend scorer config
    const scorerConfig = transformScheduledScorer(scheduledScorer);

    // Return the serialized scorer
    return scorerConfig.serialized_scorer;
  }, [getValues]);
};

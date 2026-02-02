import React, { useState } from 'react';
import { FormProvider, useForm, useWatch } from 'react-hook-form';
import { isRunningScorersEnabled } from '../../../common/utils/FeatureUtils';
import { useUpdateScheduledScorerMutation } from './hooks/useUpdateScheduledScorer';
import { convertFormDataToScheduledScorer, type ScorerFormData } from './utils/scorerTransformUtils';
import { getFormValuesFromScorer } from './scorerCardUtils';
import ScorerFormRenderer from './ScorerFormRenderer';
import type { ScheduledScorer } from './types';
import { SCORER_FORM_MODE, ScorerEvaluationScope } from './constants';

interface ScorerFormEditContainerProps {
  experimentId: string;
  onClose: () => void;
  existingScorer: ScheduledScorer;
}

const ScorerFormEditContainer: React.FC<ScorerFormEditContainerProps> = ({ experimentId, onClose, existingScorer }) => {
  // Local error state for synchronous errors
  const [componentError, setComponentError] = useState<string | null>(null);

  // Check if running scorers feature is enabled
  const isRunningScorersFeatureEnabled = isRunningScorersEnabled();

  // Hook for updating scorer
  const updateScorerMutation = useUpdateScheduledScorerMutation();

  const form = useForm<ScorerFormData>({
    mode: 'onChange', // Enable real-time validation
    defaultValues: getFormValuesFromScorer(existingScorer),
  });

  const {
    handleSubmit,
    control,
    reset,
    setValue,
    getValues,
    formState: { isValid, isDirty },
  } = form;

  // Watch the scorer type from form data
  const scorerType = useWatch({ control, name: 'scorerType' });

  const onFormSubmit = (data: ScorerFormData) => {
    try {
      setComponentError(null);

      // Convert form data to ScheduledScorer - this could throw synchronously
      const scheduledScorer = convertFormDataToScheduledScorer(data, existingScorer);

      // Update existing scorer
      updateScorerMutation.mutate(
        {
          experimentId,
          scheduledScorers: [scheduledScorer],
        },
        {
          onSuccess: () => {
            setComponentError(null);
            onClose();
            reset();
          },
          onError: () => {
            // Keep form open when there's an error so user can see error message and retry
          },
        },
      );
    } catch (error: any) {
      setComponentError(error?.message || error?.displayMessage || 'Failed to update scorer');
    }
  };

  const handleCancel = () => {
    onClose();
    reset();
    setComponentError(null); // Clear local error state
    updateScorerMutation.reset(); // Clear mutation error state
  };

  const isSubmitDisabled = updateScorerMutation.isLoading || scorerType === 'custom-code' || !isDirty || !isValid;

  return (
    <div
      css={{
        ...(isRunningScorersFeatureEnabled ? { height: '100%' } : { maxHeight: '70vh' }),
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <FormProvider {...form}>
        <ScorerFormRenderer
          mode={SCORER_FORM_MODE.EDIT}
          handleSubmit={handleSubmit}
          onFormSubmit={onFormSubmit}
          control={control}
          setValue={setValue}
          getValues={getValues}
          scorerType={scorerType}
          mutation={updateScorerMutation}
          componentError={componentError}
          handleCancel={handleCancel}
          isSubmitDisabled={isSubmitDisabled}
          experimentId={experimentId}
        />
      </FormProvider>
    </div>
  );
};

export default ScorerFormEditContainer;

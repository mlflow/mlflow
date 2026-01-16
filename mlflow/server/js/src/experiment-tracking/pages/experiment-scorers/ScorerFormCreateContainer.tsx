import React, { useState } from 'react';
import { FormProvider, useForm, useWatch } from 'react-hook-form';
import { isRunningScorersEnabled } from '../../../common/utils/FeatureUtils';
import { useCreateScheduledScorerMutation } from './hooks/useCreateScheduledScorer';
import { convertFormDataToScheduledScorer, type ScorerFormData } from './utils/scorerTransformUtils';
import ScorerFormRenderer from './ScorerFormRenderer';
import { SCORER_FORM_MODE, ScorerEvaluationScope } from './constants';

interface ScorerFormCreateContainerProps {
  experimentId: string;
  onClose: () => void;
  initialScorerType?: ScorerFormData['scorerType'];
}

const ScorerFormCreateContainer: React.FC<ScorerFormCreateContainerProps> = ({
  experimentId,
  onClose,
  initialScorerType = 'llm',
}) => {
  // Local error state for synchronous errors
  const [componentError, setComponentError] = useState<string | null>(null);

  // Check if running scorers feature is enabled
  const isRunningScorersFeatureEnabled = isRunningScorersEnabled();

  // Hook for creating scorer
  const createScorerMutation = useCreateScheduledScorerMutation();

  const form = useForm<ScorerFormData>({
    mode: 'onChange', // Enable real-time validation
    defaultValues: {
      scorerType: initialScorerType,
      name: '',
      sampleRate: 100,
      filterString: '',
      llmTemplate: 'Custom',
      model: '',
      disableMonitoring: false,
      isInstructionsJudge: true, // Custom template is an instructions judge
      evaluationScope: ScorerEvaluationScope.TRACES,
    },
  });

  const {
    handleSubmit,
    control,
    reset,
    setValue,
    getValues,
    formState: { isValid },
  } = form;

  // Watch the scorer type from form data
  const scorerType = useWatch({ control, name: 'scorerType' });

  const onFormSubmit = (data: ScorerFormData) => {
    try {
      setComponentError(null);

      // Convert form data to ScheduledScorer - this could throw synchronously
      const scheduledScorer = convertFormDataToScheduledScorer(data, undefined);

      // Create new scorer
      createScorerMutation.mutate(
        {
          experimentId,
          scheduledScorer,
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
      setComponentError(error?.message || error?.displayMessage || 'Failed to create scorer');
    }
  };

  const handleCancel = () => {
    onClose();
    reset();
    setComponentError(null); // Clear local error state
    createScorerMutation.reset(); // Clear mutation error state
  };

  const isSubmitDisabled = createScorerMutation.isLoading || scorerType === 'custom-code' || !isValid;

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
          mode={SCORER_FORM_MODE.CREATE}
          handleSubmit={handleSubmit}
          onFormSubmit={onFormSubmit}
          control={control}
          setValue={setValue}
          getValues={getValues}
          scorerType={scorerType}
          mutation={createScorerMutation}
          componentError={componentError}
          handleCancel={handleCancel}
          isSubmitDisabled={isSubmitDisabled}
          experimentId={experimentId}
        />
      </FormProvider>
    </div>
  );
};

export default ScorerFormCreateContainer;

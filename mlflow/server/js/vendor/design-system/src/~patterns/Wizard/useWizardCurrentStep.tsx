import { useCallback, useMemo, useState } from 'react';

export interface WizardCurrentStepParams {
  currentStepIndex: number;
  setCurrentStepIndex: (step: number) => void;
  totalSteps: number;
  onValidateStepChange?: (step: number) => Promise<boolean>;
  onStepChanged: ({ step, completed }: { step: number; completed: boolean }) => void;
}

export function useWizardCurrentStep({
  currentStepIndex,
  setCurrentStepIndex,
  totalSteps,
  onValidateStepChange,
  onStepChanged,
}: WizardCurrentStepParams) {
  const [busyValidatingNextStep, setBusyValidatingNextStep] = useState<boolean>(false);
  const isLastStep = useMemo(() => currentStepIndex === totalSteps - 1, [currentStepIndex, totalSteps]);

  const onStepsChange = useCallback(
    async (step: number, completed = false) => {
      if (!completed && step === currentStepIndex) return;

      setCurrentStepIndex(step);
      onStepChanged({ step, completed });
    },
    [currentStepIndex, onStepChanged, setCurrentStepIndex],
  );

  const goToNextStepOrDone = useCallback(async () => {
    if (onValidateStepChange) {
      setBusyValidatingNextStep(true);
      try {
        const approvedStepChange = await onValidateStepChange(currentStepIndex);
        if (!approvedStepChange) {
          return;
        }
      } finally {
        setBusyValidatingNextStep(false);
      }
    }

    onStepsChange(Math.min(currentStepIndex + 1, totalSteps - 1), isLastStep);
  }, [currentStepIndex, isLastStep, onStepsChange, onValidateStepChange, totalSteps]);

  const goToPreviousStep = useCallback(() => {
    if (currentStepIndex > 0) {
      onStepsChange(currentStepIndex - 1);
    }
  }, [currentStepIndex, onStepsChange]);

  const goToStep = useCallback(
    (step: number) => {
      if (step > -1 && step < totalSteps) {
        onStepsChange(step);
      }
    },
    [onStepsChange, totalSteps],
  );

  return {
    busyValidatingNextStep,
    isLastStep,
    onStepsChange,
    goToNextStepOrDone,
    goToPreviousStep,
    goToStep,
  };
}

export type WizardCurrentStepResult = ReturnType<typeof useWizardCurrentStep>;

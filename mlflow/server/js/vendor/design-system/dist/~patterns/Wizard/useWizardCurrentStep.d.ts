export interface WizardCurrentStepParams {
    currentStepIndex: number;
    setCurrentStepIndex: (step: number) => void;
    totalSteps: number;
    onValidateStepChange?: (step: number) => Promise<boolean>;
    onStepChanged: ({ step, completed }: {
        step: number;
        completed: boolean;
    }) => void;
}
export declare function useWizardCurrentStep({ currentStepIndex, setCurrentStepIndex, totalSteps, onValidateStepChange, onStepChanged, }: WizardCurrentStepParams): {
    busyValidatingNextStep: boolean;
    isLastStep: boolean;
    onStepsChange: (step: number, completed?: any) => Promise<void>;
    goToNextStepOrDone: () => Promise<void>;
    goToPreviousStep: () => void;
    goToStep: (step: number) => void;
};
export type WizardCurrentStepResult = ReturnType<typeof useWizardCurrentStep>;
//# sourceMappingURL=useWizardCurrentStep.d.ts.map
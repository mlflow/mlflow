import type { WizardStep } from './WizardStep';
export declare function useStepperStepsFromWizardSteps(wizardSteps: WizardStep[], currentStepIdx: number, hideDescriptionForFutureSteps: boolean): {
    title: import("react").ReactNode;
    status?: "completed" | "error" | "loading" | "upcoming" | "warning" | undefined;
    description: import("react").ReactNode;
    additionalVerticalContent: import("react").ReactNode;
    clickEnabled: boolean;
}[];
export declare function isWizardStepEnabled(steps: WizardStep[], stepIdx: number, currentStepIdx: number, status: WizardStep['status']): boolean;
//# sourceMappingURL=useStepperStepsFromWizardSteps.d.ts.map
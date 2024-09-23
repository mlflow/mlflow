import type { WizardStep } from './WizardStep';
export declare function useStepperStepsFromWizardSteps(wizardSteps: WizardStep[], currentStepIdx: number): {
    additionalVerticalContent: import("react").ReactNode;
    clickEnabled: boolean;
    title: React.ReactNode;
    status?: import("../../development").Step["status"];
    description?: React.ReactNode;
}[];
export declare function isWizardStepEnabled(steps: WizardStep[], stepIdx: number, currentStepIdx: number, status: WizardStep['status']): boolean;
//# sourceMappingURL=useStepperStepsFromWizardSteps.d.ts.map
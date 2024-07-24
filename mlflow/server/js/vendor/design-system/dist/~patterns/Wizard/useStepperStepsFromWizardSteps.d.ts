import type { WizardStep } from './WizardStep';
export declare function useStepperStepsFromWizardSteps(wizardSteps: WizardStep[], currentStepIdx: number): {
    additionalVerticalContent: import("react").ReactNode;
    clickEnabled: boolean;
    title: React.ReactNode;
    status?: import("../../development").Step["status"];
    description?: React.ReactNode;
}[];
//# sourceMappingURL=useStepperStepsFromWizardSteps.d.ts.map
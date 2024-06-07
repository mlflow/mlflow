/// <reference types="react" />
import type { WizardStep } from './WizardStep';
export declare function useStepperStepsFromWizardSteps(wizardSteps: WizardStep[], currentStepIdx: number): {
    additionalVerticalContent: import("react").ReactNode;
    clickEnabled: boolean;
    title: import("react").ReactNode;
    status?: "error" | "warning" | "loading" | "completed" | "upcoming" | undefined;
    description?: import("react").ReactNode;
}[];
//# sourceMappingURL=useStepperStepsFromWizardSteps.d.ts.map
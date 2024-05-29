/// <reference types="react" />
import type { WizardStep } from './WizardStep';
export declare function useStepperStepsFromWizardSteps(wizardSteps: WizardStep[]): {
    additionalVerticalContent: import("react").ReactNode;
    title: import("react").ReactNode;
    status?: "error" | "loading" | "completed" | "upcoming" | undefined;
    description?: import("react").ReactNode;
}[];
//# sourceMappingURL=useStepperStepsFromWizardSteps.d.ts.map
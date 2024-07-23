import type { WizardStep } from './WizardStep';
import type { StepperProps } from '../../development/Stepper';
export interface HorizontalWizardStepsContentProps {
    steps: WizardStep[];
    currentStepIndex: number;
    localizeStepNumber: StepperProps['localizeStepNumber'];
    enableClickingToSteps: boolean;
    goToStep: (step: number) => void;
}
export declare function HorizontalWizardStepsContent({ steps: wizardSteps, currentStepIndex, localizeStepNumber, enableClickingToSteps, goToStep, }: HorizontalWizardStepsContentProps): import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=HorizontalWizardStepsContent.d.ts.map
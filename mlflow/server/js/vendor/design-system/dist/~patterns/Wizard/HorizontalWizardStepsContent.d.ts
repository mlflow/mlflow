import type { WizardStep } from './WizardStep';
import type { StepperProps } from '../../development/Stepper';
export interface HorizontalWizardStepsContentProps {
    steps: WizardStep[];
    currentStepIndex: number;
    localizeStepNumber: StepperProps['localizeStepNumber'];
    enableClickingToSteps: boolean;
    goToStep: (step: number) => void;
    hideDescriptionForFutureSteps?: boolean;
}
export declare function HorizontalWizardStepsContent({ steps: wizardSteps, currentStepIndex, localizeStepNumber, enableClickingToSteps, goToStep, hideDescriptionForFutureSteps, }: HorizontalWizardStepsContentProps): import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=HorizontalWizardStepsContent.d.ts.map
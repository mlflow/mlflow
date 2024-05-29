import type { WizardStep } from './WizardStep';
import type { StepperProps } from '../../development/Stepper';
export interface HorizontalWizardStepsContentProps {
    steps: WizardStep[];
    currentStepIndex: number;
    expandContentToFullHeight: boolean;
    localizeStepNumber: StepperProps['localizeStepNumber'];
    useCustomScrollBehavior?: boolean;
}
export declare function HorizontalWizardStepsContent({ steps: wizardSteps, currentStepIndex, expandContentToFullHeight, useCustomScrollBehavior, localizeStepNumber, }: HorizontalWizardStepsContentProps): import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=HorizontalWizardStepsContent.d.ts.map
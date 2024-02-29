import type { WizardStep } from './WizardStep';
import type { StepperProps } from '../../development/Stepper';
interface WizardStepsContentProps {
    steps: WizardStep[];
    currentStepIndex: number;
    expandContentToFullHeight: boolean;
    localizeStepNumber: StepperProps['localizeStepNumber'];
}
export declare function WizardStepsContent({ steps: wizardSteps, currentStepIndex, expandContentToFullHeight, localizeStepNumber, }: WizardStepsContentProps): import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=WizardStepsContent.d.ts.map
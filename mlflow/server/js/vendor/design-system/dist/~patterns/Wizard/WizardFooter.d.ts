import type { WizardStep } from './WizardStep';
import type { WizardCurrentStepResult } from './useWizardCurrentStep';
import { type ButtonProps } from '../../design-system';
export interface WizardFooterProps extends Omit<WizardCurrentStepResult, 'onStepsChange' | 'onValidateStepChange'>, WizardStep {
    currentStepIndex: number;
    onCancel?: () => void;
    moveCancelToOtherSide?: boolean;
    cancelButtonContent: React.ReactNode;
    nextButtonContent: React.ReactNode;
    previousButtonContent: React.ReactNode;
    doneButtonContent: React.ReactNode;
    extraFooterButtonsLeft?: ButtonProps[];
    extraFooterButtonsRight?: ButtonProps[];
}
export declare function getWizardFooterButtons({ goToNextStepOrDone, isLastStep, currentStepIndex, goToPreviousStep, busyValidatingNextStep, nextButtonDisabled, nextButtonLoading, nextButtonContentOverride, previousButtonContentOverride, previousStepButtonHidden, previousButtonDisabled, previousButtonLoading, cancelButtonContent, cancelStepButtonHidden, nextButtonContent, previousButtonContent, doneButtonContent, extraFooterButtonsLeft, extraFooterButtonsRight, onCancel, moveCancelToOtherSide, }: WizardFooterProps): (import("@emotion/react/jsx-runtime").JSX.Element | import("@emotion/react/jsx-runtime").JSX.Element[])[];
//# sourceMappingURL=WizardFooter.d.ts.map
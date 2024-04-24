/// <reference types="react" />
import type { WizardStep } from './WizardStep';
import type { WizardCurrentStepResult } from './useWizardCurrentStep';
import type { ButtonProps } from '..';
export interface WizardFooterProps extends Omit<WizardCurrentStepResult, 'onStepsChange' | 'onValidateStepChange'>, WizardStep {
    onCancel?: () => void;
    cancelButtonContent: React.ReactNode;
    nextButtonContent: React.ReactNode;
    previousButtonContent: React.ReactNode;
    doneButtonContent: React.ReactNode;
    extraFooterButtons?: ButtonProps[];
}
export declare function WizardFooter(props: WizardFooterProps): import("@emotion/react/jsx-runtime").JSX.Element;
export declare function getWizardFooterButtons({ goToNextStepOrDone, isLastStep, currentStepIndex, goToPreviousStep, busyValidatingNextStep, nextButtonDisabled, nextButtonLoading, nextButtonContentOverride, previousButtonContentOverride, previousStepButtonHidden, previousButtonDisabled, previousButtonLoading, cancelButtonContent, cancelStepButtonHidden, nextButtonContent, previousButtonContent, doneButtonContent, extraFooterButtons, onCancel, }: WizardFooterProps): (import("@emotion/react/jsx-runtime").JSX.Element | import("@emotion/react/jsx-runtime").JSX.Element[])[];
//# sourceMappingURL=WizardFooter.d.ts.map
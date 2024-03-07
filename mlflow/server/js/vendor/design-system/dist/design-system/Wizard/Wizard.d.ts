/// <reference types="react" />
import type { WizardStep } from './WizardStep';
import type { WizardCurrentStepParams, WizardCurrentStepResult } from './useWizardCurrentStep';
import type { ButtonProps } from '..';
import type { StepperProps } from '../../development/Stepper';
export interface WizardProps {
    /**
     * The steps in this wizard.
     * If steps are empty, this component won't render anything
     */
    steps: WizardStep[];
    /**
     * Fired when the step has changed. Completed will be true when clicking done on the final step
     */
    onStepChanged: WizardCurrentStepParams['onStepChanged'];
    /**
     * Fired when clicking on next and if the promise returns false, it will not advance to the next step
     * This is useful for validating fields when users click on next and if invalid keep the user on this step
     */
    onValidateStepChange?: WizardCurrentStepParams['onValidateStepChange'];
    /**
     * Function to localize the step number; workaround for no react-intl support within dubois
     *
     * ex) localizeStepNumber={intl.formatNumber}
     */
    localizeStepNumber: StepperProps['localizeStepNumber'];
    /**
     * Called when user clicks on cancel button in Wizard footer
     */
    onCancel: () => void;
    /**
     * Initial step of the wizard
     *
     * @default 0
     */
    initialStep?: number;
    /**
     * Width of the entire component
     *
     * @default 100%
     */
    width?: string | number;
    /**
     * height of the entire component
     *
     * @default 100%
     */
    height?: string | number;
    /**
     * Content of the wizard's footer cancel button
     */
    cancelButtonContent: React.ReactNode;
    /**
     * Content of the wizard's footer next button
     *
     * Can be overriden for a specific step by setting `nextButtonContentOverride` on the step
     */
    nextButtonContent: React.ReactNode;
    /**
     * Content of the wizard's footer previous button
     *
     * Can be overriden for a specific step by setting `previousButtonContentOverride` on the step
     */
    previousButtonContent: React.ReactNode;
    /**
     * Content of the wizard's footer done (last step's next) button
     *
     * Can be overriden for a specific step by setting `nextButtonContentOverride` on the last step
     */
    doneButtonContent: React.ReactNode;
    /**
     * Extra set of buttons to be displayed in the footer to the right of the cancel button
     * The only button property that will be overriden is type to default; keeping 'next' as the only primary button.
     */
    extraFooterButtons?: ButtonProps[];
    /**
     * If true the content of the wizard will take up all available vertical space.
     * This is to keep the footer at the bottom of the wizard
     *
     * A height on either the wizard parent or using the height prop is required for this to work
     *
     * @default true
     */
    expandContentToFullHeight?: boolean;
}
export declare function Wizard({ steps, onStepChanged, onValidateStepChange, initialStep, ...props }: WizardProps): import("@emotion/react/jsx-runtime").JSX.Element;
export type WizardControlledProps = WizardProps & WizardCurrentStepResult;
export declare function WizardControlled({ initialStep, steps, width, height, currentStepIndex, localizeStepNumber, onStepsChange, isLastStep, expandContentToFullHeight, ...footerProps }: WizardControlledProps): import("@emotion/react/jsx-runtime").JSX.Element | null;
//# sourceMappingURL=Wizard.d.ts.map
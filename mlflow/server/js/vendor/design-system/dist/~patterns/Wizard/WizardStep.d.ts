import type { Step } from '../../development/Stepper';
/**
 * Represents one step in the wizard
 */
export interface WizardStep {
    /**
     * The title of the step displayed in the steps header
     */
    title: React.ReactNode;
    /**
     * Content of the step; the actual form
     */
    content: React.ReactNode;
    /**
     * Optional description of the step displayed in the steps header
     */
    description?: React.ReactNode;
    /**
     * if true the step is completed, reflected in the stepper style
     */
    status?: Step['status'];
    /**
     * if true the next footer button for this step will be disabled
     *
     * @default false
     */
    nextButtonDisabled: boolean;
    /**
     * if true the next footer button for this step will be loading
     *
     * @default false
     */
    nextButtonLoading?: boolean;
    /**
     * Override content of this step's next button
     *
     * ex) 'Next', 'Create', 'Done' (for the last step)
     */
    nextButtonContentOverride?: React.ReactNode;
    /**
     * If true the previous button will be hidden for this step.
     * Note for the first step this is a no-op as it will always be hidden
     *
     * @default false
     */
    previousStepButtonHidden?: boolean;
    /**
     * Override content of this step's previous button
     */
    previousButtonContentOverride?: React.ReactNode;
    /**
     * if true the previous footer button for this step will be disabled
     *
     * @default false
     */
    previousButtonDisabled?: boolean;
    /**
     * if true the previous footer button for this step will be loading
     *
     * @default false
     */
    previousButtonLoading?: boolean;
    /**
     * If true the cancel button will be hidden for this step.
     *
     * @default false
     */
    cancelStepButtonHidden?: boolean;
    /**
     * Additional content displayed beneath the step description of the stepper in a horizontal Wizard layout
     */
    additionalHorizontalLayoutStepContent?: Step['additionalVerticalContent'];
    /**
     * If true the step content will take up all available vertical space.
     * This is to keep the footer at the bottom of the wizard
     *
     * A height on either the wizard parent or using the height prop is required for this to work
     *
     * @default true
     */
    expandContentToFullHeight?: boolean;
    /**
     * Delegates all content scroll behavior to the caller if true
     *    Disable the default scroll drop shadow
     *    Hide the step content overflow
     * @default false
     */
    disableDefaultScrollBehavior?: boolean;
    /**
     * Enable this step to be clicked
     * The wizard must have property `enableClickingToSteps` set to true for this to work
     */
    clickEnabled?: boolean;
}
//# sourceMappingURL=WizardStep.d.ts.map
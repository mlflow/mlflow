/// <reference types="react" />
import type { Step } from '../../development/Stepper';
/**
 * Represents one step in the wizard
 */
export interface WizardStep {
    /**
     * The title of the step displayed in the steps header
     */
    title: string;
    /**
     * Content of the step; the actual form
     */
    content: React.ReactNode;
    /**
     * Optional description of the step displayed in the steps header
     */
    description?: string;
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
}
//# sourceMappingURL=WizardStep.d.ts.map
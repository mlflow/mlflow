import type { ButtonProps } from '@databricks/design-system';
import type { WizardStep } from './WizardStep';
import type { WizardCurrentStepParams, WizardCurrentStepResult } from './useWizardCurrentStep';
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
     * Button content for the compact vertical wizard's stepper's trigger.
     * In small screen widths we hide the vertical stepper and display a popover button instead
     *
     * This is required to enable the compact vertical wizard layout.
     * Ex) (currentStepIndex, totalSteps) =>
     *       intl.formatMessage( { defaultMessage: 'Step {currentStepIndex} / {totalSteps}', description: '', }, { currentStepIndex: currentStepIndex + 1, totalSteps })
     */
    verticalCompactButtonContent?: (currentStepIndex: number, totalSteps: number) => string;
    /**
     * Configuration to render a DocumentationSidebar to the right of the vertical wizard's step content
     *
     * content: will be used as the `DocumentationSidebar.Content`. The Content child component takes in a `contentId: string` property; this is the contentId passed along from the
  Trigger. This allows the client to display different help based on the contentId trigger clicked.
      *
      * title: is title displayed atop of the `DocumentationSidebar.Content`
      *
      * modalTitleWhenCompact: is the modal title for the compact version of the DocumentationSidebar
      *
      * closeLabel: is the aria label used for the sidebar close button
     */
    verticalDocumentationSidebarConfig?: {
        content: React.ReactNode;
        title: string;
        modalTitleWhenCompact?: string;
        closeLabel: string;
    };
    /**
     * Called when user clicks on cancel button in Wizard footer
     */
    onCancel: () => void;
    /**
     * Initial step of the wizard
     * If this is an invalid step index; no wizard will be rendered
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
     * Extra set of ordered buttons to be displayed in the footer to the left of the next button
     * The only button property that will be overriden is type to default; keeping the far right button as the only primary button.
     */
    extraFooterButtonsLeft?: ButtonProps[];
    /**
     * Extra set of ordered buttons to be displayed in the footer to the right of the next button
     * This will make the next button a default button and the last button in this list as primary
     */
    extraFooterButtonsRight?: ButtonProps[];
    /**
     * Layout of the stepper.
     * A vertical wizard will have a vertical stepper on the left side of the step content
     * A horizontal wizard will have a horizontal stepper atop of the step content.
     * Note this is here for historical reasons; vertical layouts are highly recommended
     *
     * @default 'vertical'
     */
    layout?: 'horizontal' | 'vertical';
    /**
     * Optional title of the wizard displayed above the stepper in the horizontal layout
     */
    title?: React.ReactNode;
    /**
     * If not set, the default padding will be applied to the wizard
     *
     * @default theme.spacing.lg
     */
    padding?: string | number;
    /**
     * If true user will be able to click to steps that have been completed, in error or warning states, or are less than current step,
     * or every step before a step is completed.
     * This default behavior can be overriden by setting `clickEnabled` on each `WizardStep`
     *
     * @default false
     */
    enableClickingToSteps?: boolean;
}
export type WizardControlledProps = WizardProps & WizardCurrentStepResult & {
    currentStepIndex: number;
};
//# sourceMappingURL=WizardProps.d.ts.map
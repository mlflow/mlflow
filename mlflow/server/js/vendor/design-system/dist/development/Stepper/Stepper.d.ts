export interface Step {
    /**
     * Title of the step
     */
    title: React.ReactNode;
    /**
     * Optional description of the step
     */
    description?: React.ReactNode;
    /**
     * Status of the step. This will change the icon and text color of the step.
     *
     * @default 'upcoming'
     */
    status?: 'completed' | 'loading' | 'upcoming' | 'error' | 'warning';
    /**
     * Custom icon to display in the step. If provided, the `icon` prop will be used instead of the default icon.
     */
    icon?: React.ComponentType<{
        statusColor: string;
        status: Step['status'];
    }>;
    /**
     * Additional content displayed beneath the step description a vertical stepper
     *
     * This can be used to create a vertical wizard
     */
    additionalVerticalContent?: React.ReactNode;
    /**
     * If true, the step can be clicked and the `onStepClicked` callback will be called
     */
    clickEnabled?: boolean;
}
export interface StepperProps {
    /**
     * List of ordered steps in the stepper
     */
    steps: Step[];
    /**
     * Function to localize the step number; workaround for no react-intl support within dubois
     *
     * ex) localizeStepNumber={intl.formatNumber}
     */
    localizeStepNumber: (stepIndex: number) => string;
    /**
     * Direction of the stepper
     *
     * @default horizontal
     */
    direction?: 'horizontal' | 'vertical';
    /**
     * Current active step from the `steps` property (zero-indexed)
     *
     * @default 0
     */
    currentStepIndex?: number;
    /**
     * If true, and the stepper has a horizontal direction the stepper will be updated to be vertical if width is less than 532px.
     * Set this value to false to opt out of the responsive behavior.
     *
     * @default true
     */
    responsive?: boolean;
    /**
     * Callback when a step is clicked for steps with `clickEnabled` set to true
     *
     * @default 'undefined'
     */
    onStepClicked?: (stepIndex: number) => void;
}
export declare function Stepper({ direction: requestedDirection, currentStepIndex: currentStep, steps, localizeStepNumber, responsive, onStepClicked, }: StepperProps): import("@emotion/react/jsx-runtime").JSX.Element | null;
export declare function useResponsiveDirection({ requestedDirection, responsive, enabled, ref, }: {
    requestedDirection: StepperProps['direction'];
    enabled: boolean;
    responsive: boolean;
    ref: React.RefObject<HTMLOListElement>;
}): {
    direction: "horizontal" | "vertical";
};
//# sourceMappingURL=Stepper.d.ts.map
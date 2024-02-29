/// <reference types="react" />
export interface Step {
    /**
     * Title of the step
     */
    title: string;
    /**
     * Optional description of the step
     */
    description?: string;
    /**
     * Status of the step. This will change the icon and text color of the step.
     *
     * @default 'upcoming'
     */
    status?: 'completed' | 'loading' | 'upcoming' | 'error';
    icon?: React.ComponentType<{
        statusColor: string;
        status: Step['status'];
    }>;
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
}
export declare function Stepper({ direction: requestedDirection, currentStepIndex: currentStep, steps, localizeStepNumber, responsive, }: StepperProps): import("@emotion/react/jsx-runtime").JSX.Element | null;
export declare function useResponsiveDirection({ requestedDirection, responsive, enabled, ref, }: {
    requestedDirection: StepperProps['direction'];
    enabled: boolean;
    responsive: boolean;
    ref: React.RefObject<HTMLOListElement>;
}): {
    direction: "horizontal" | "vertical";
};
//# sourceMappingURL=Stepper.d.ts.map
import type { WizardControlledProps } from './WizardProps';
type VerticalWizardContentProps = Omit<WizardControlledProps, 'layout' | 'initialStep'>;
export declare function VerticalWizardContent({ width, height, steps: wizardSteps, currentStepIndex, localizeStepNumber, onStepsChange, title, padding, verticalCompactButtonContent, enableClickingToSteps, ...footerProps }: VerticalWizardContentProps): import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=VerticalWizardContent.d.ts.map
import type { WizardControlledProps } from './WizardProps';
type HorizontalWizardContentProps = Omit<WizardControlledProps, 'layout' | 'title' | 'initialStep'>;
export declare function HorizontalWizardContent({ width, height, steps, currentStepIndex, localizeStepNumber, onStepsChange, enableClickingToSteps, ...footerProps }: HorizontalWizardContentProps): import("@emotion/react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=HorizontalWizardContent.d.ts.map